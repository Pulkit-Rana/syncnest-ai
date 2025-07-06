import os
import requests
from dotenv import load_dotenv

from agent.vector.qdrant_client import add_documents, init_qdrant

load_dotenv()

# ======= ADO CONFIG =======
ADO_ORG = os.getenv("ADO_ORGANIZATION")
ADO_PROJECT = os.getenv("ADO_PROJECT")
ADO_PAT = os.getenv("ADO_PAT")

API_BASE = f"https://dev.azure.com/{ADO_ORG}/{ADO_PROJECT}/_apis"
HEADERS = {"Content-Type": "application/json"}
AUTH = ("", ADO_PAT)  # PAT as username (blank password)

TOP_K = 200  # Index this many items per type (customize as needed)

# ======= Qdrant Setup =======
init_qdrant()  # Only creates collection if not exists

def fetch_work_items(types=("User Story", "Feature", "Bug"), max_items=TOP_K):
    wiql_types = " OR ".join([f"[System.WorkItemType] = '{t}'" for t in types])
    wiql = {
        "query": f"""
        SELECT [System.Id], [System.Title], [System.Description], [System.WorkItemType]
        FROM WorkItems
        WHERE ({wiql_types})
        ORDER BY [System.ChangedDate] DESC
        """
    }
    url = f"{API_BASE}/wit/wiql?api-version=7.1-preview.2"
    resp = requests.post(url, auth=AUTH, headers=HEADERS, json=wiql, timeout=10)
    print("WORK ITEMS STATUS:", resp.status_code)
    work_items = resp.json().get("workItems", []) if resp.status_code == 200 else []
    ids = [str(wi["id"]) for wi in work_items[:max_items]]

    if not ids:
        return []
    details_url = f"{API_BASE}/wit/workitems?ids={','.join(ids)}&$expand=fields&api-version=7.1-preview.3"
    details_resp = requests.get(details_url, auth=AUTH, headers=HEADERS, timeout=10)
    print("WORK ITEM DETAILS STATUS:", details_resp.status_code)
    items = []
    for wi in details_resp.json().get("value", []):
        fields = wi.get("fields", {})
        items.append({
            "id": wi.get("id"),
            "title": fields.get("System.Title", ""),
            "description": fields.get("System.Description", ""),
            "type": fields.get("System.WorkItemType", ""),
            "source": "work_item"
        })
    return items

def fetch_all_known_wiki_pages(wiki_id, max_id=10):
    items = []
    for page_id in range(1, max_id+1):  # Try IDs 1 to max_id
        content_url = f"{API_BASE}/wiki/wikis/{wiki_id}/pages/{page_id}?includeContent=True&api-version=7.1-preview.1"
        content_resp = requests.get(content_url, auth=AUTH, headers=HEADERS, timeout=10)
        print(f"FETCHING PAGE {page_id} for wiki {wiki_id}: {content_resp.status_code}")
        if content_resp.status_code != 200:
            continue  # Skip missing/invalid pages
        content = content_resp.json().get("content", "")
        # Path/title logic (fallback for home pages)
        title = content_resp.json().get("path", "").strip("/").split("/")[-1] or f"Page {page_id}"
        items.append({
            "id": f"{wiki_id}:{page_id}",
            "title": title,
            "description": content[:1500],
            "type": "Wiki",
            "source": "wiki"
        })
    return items

def fetch_wiki_pages_fixed_bruteforce(max_pages=10):
    wikis_url = f"{API_BASE}/wiki/wikis?api-version=7.1-preview.1"
    wikis_resp = requests.get(wikis_url, auth=AUTH, headers=HEADERS, timeout=10)
    print("WIKIS STATUS:", wikis_resp.status_code)
    print("WIKIS RESPONSE:", wikis_resp.json())
    items = []
    if wikis_resp.status_code == 200:
        for wiki in wikis_resp.json().get("value", []):
            wiki_id = wiki.get("id")  # This is the GUID!
            print(f"Fetching ALL pages (ID brute force) for wiki id: {wiki_id}")
            items.extend(fetch_all_known_wiki_pages(wiki_id, max_id=max_pages))
    else:
        print("Failed to fetch wikis. Response:", wikis_resp.text)
    return items

def build_docs_and_meta(items):
    docs = []
    meta = []
    for i in items:
        doc = f"{i.get('title', '')}\n{i.get('description', '')}"
        docs.append(doc)
        meta.append({
            "id": i["id"],
            "title": i.get("title", ""),
            "type": i.get("type", ""),
            "source": i.get("source", ""),
        })
    return docs, meta

if __name__ == "__main__":
    print("Fetching work items from ADO...")
    stories = fetch_work_items()
    print(f"Fetched {len(stories)} stories/bugs/features.")
    if stories:
        print("First work item meta:", stories[0])

    print("Fetching wiki pages from ADO (using brute-force by page id)...")
    # Set max_pages to the highest page_id you have in your wiki, or just try 10/20
    wikis = fetch_wiki_pages_fixed_bruteforce(max_pages=10)
    print(f"Fetched {len(wikis)} wiki pages.")
    if wikis:
        print("First wiki meta:", wikis[0])
    else:
        print("NO wiki docs fetched from ADO. Fix this first.")

    all_items = stories + wikis
    docs, meta = build_docs_and_meta(all_items)
    print(f"Indexing {len(docs)} docs to Qdrant...")
    for idx, m in enumerate(meta):
        if m["source"] == "wiki":
            print(f"WIKI META [{idx}]: {m}")
    add_documents(docs, meta)
    print("✅ Semantic index complete!")
