import os
import requests
from typing import List, Dict, Optional

class ADOClient:
    def __init__(self, organization: Optional[str] = None, project: Optional[str] = None, pat: Optional[str] = None):
        self.organization = organization or os.environ.get("ADO_ORGANIZATION")
        self.project = project or os.environ.get("ADO_PROJECT")
        self.pat = pat or os.environ.get("ADO_PAT")
        self.api_base = f"https://dev.azure.com/{self.organization}/{self.project}/_apis"
        self.headers = {"Content-Type": "application/json"}
        self.auth = ("", self.pat)  # PAT as password, blank username

    def search_stories(
        self,
        query: str,
        top_k: int = 10,
        group_by_type: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Searches ADO for user stories, features, bugs, and Wiki pages matching the query.
        Returns a dict: keys = 'bugs', 'stories', 'features', 'wikis'.
        Each list: dicts with fields: id, title, description, status, work_item_type, last_modified, source.
        """
        results = {
            "bugs": [],
            "stories": [],
            "features": [],
            "wikis": []
        }

        # ---- 1. Work Items (Bugs, Stories, Features) ----
        attempts = [query.strip()]
        lowered = query.lower()
        # Try common issue-related keywords if not found
        for keyword in ["not working", "error", "fails", "issue", "bug", "filter", "button"]:
            if keyword in lowered and keyword not in attempts:
                attempts.append(keyword)

        for attempt in attempts:
            wiql = {
                "query": f"""
                SELECT [System.Id], [System.Title], [System.Description], [System.WorkItemType], [System.State], [System.ChangedDate]
                FROM WorkItems
                WHERE
                    ([System.WorkItemType] = 'User Story' OR [System.WorkItemType] = 'Feature' OR [System.WorkItemType] = 'Bug')
                    AND ([System.Title] CONTAINS '{attempt}' OR [System.Description] CONTAINS '{attempt}')
                ORDER BY [System.ChangedDate] DESC
                """
            }

            url = f"{self.api_base}/wit/wiql?api-version=7.1-preview.2"
            try:
                resp = requests.post(url, auth=self.auth, headers=self.headers, json=wiql, timeout=10)
            except Exception as ex:
                print(f"[ADOClient] WIQL POST failed: {ex}")
                continue

            if resp.status_code != 200:
                print(f"[ADOClient] WIQL POST failed, code={resp.status_code}, text={resp.text}")
                continue

            result = resp.json()
            work_items = result.get("workItems", [])
            if not work_items:
                continue

            ids = [str(item["id"]) for item in work_items[:top_k]]
            if not ids:
                continue

            details_url = f"{self.api_base}/wit/workitems?ids={','.join(ids)}&$expand=fields&api-version=7.1-preview.3"
            try:
                details_resp = requests.get(details_url, auth=self.auth, headers=self.headers, timeout=10)
            except Exception as ex:
                print(f"[ADOClient] Details GET failed: {ex}")
                continue

            if details_resp.status_code != 200:
                print(f"[ADOClient] Details GET failed, code={details_resp.status_code}, text={details_resp.text}")
                continue

            for wi in details_resp.json().get("value", []):
                fields = wi.get("fields", {})
                item = {
                    "id": wi.get("id"),
                    "title": fields.get("System.Title", ""),
                    "description": fields.get("System.Description", ""),
                    "status": fields.get("System.State", ""),
                    "work_item_type": fields.get("System.WorkItemType", ""),
                    "last_modified": fields.get("System.ChangedDate", ""),
                    "source": "work_item"
                }
                wtype = item["work_item_type"].lower()
                if "bug" in wtype:
                    results["bugs"].append(item)
                elif "story" in wtype:
                    results["stories"].append(item)
                elif "feature" in wtype:
                    results["features"].append(item)

        # ---- 2. Wiki Search ----
        wikis_url = f"{self.api_base}/wiki/wikis?api-version=7.1-preview.1"
        try:
            wikis_resp = requests.get(wikis_url, auth=self.auth, headers=self.headers, timeout=10)
        except Exception as ex:
            print(f"[ADOClient] Wikis GET failed: {ex}")
            wikis_resp = None

        if wikis_resp and wikis_resp.status_code == 200:
            for wiki in wikis_resp.json().get("value", []):
                wiki_id = wiki.get("id")
                pages_url = f"{self.api_base}/wiki/wikis/{wiki_id}/pages?api-version=7.1-preview.1"
                try:
                    pages_resp = requests.get(pages_url, auth=self.auth, headers=self.headers, timeout=10)
                except Exception as ex:
                    print(f"[ADOClient] Wiki pages GET failed: {ex}")
                    continue
                if pages_resp.status_code != 200:
                    continue
                for page in pages_resp.json().get("value", []):
                    page_id = page.get("id")
                    title = page.get("path", "").strip("/").split("/")[-1]
                    content_url = f"{self.api_base}/wiki/wikis/{wiki_id}/pages/{page_id}?includeContent=True&api-version=7.1-preview.1"
                    try:
                        content_resp = requests.get(content_url, auth=self.auth, headers=self.headers, timeout=10)
                    except Exception as ex:
                        print(f"[ADOClient] Wiki content GET failed: {ex}")
                        continue
                    if content_resp.status_code != 200:
                        continue
                    content = content_resp.json().get("content", "")
                    # Only add if keyword matched in title/content
                    for attempt in attempts:
                        if (attempt.lower() in title.lower()) or (attempt.lower() in content.lower()):
                            results["wikis"].append({
                                "id": f"{wiki_id}:{page_id}",
                                "title": title,
                                "description": content[:500],
                                "source": "wiki"
                            })
                            break  # Only once per attempt

        for key in results:
            results[key] = results[key][:top_k]

        return results

    def create_work_item(
        self,
        work_item_type: str,
        fields: Dict[str, str]
    ) -> Dict:
        """
        Creates a new work item (Bug/User Story/Feature) in ADO using JSON-Patch.
        Returns: {id, title, url}
        """
        url = f"{self.api_base}/wit/workitems/${work_item_type}?api-version=7.1-preview.3"
        patch = []
        for k, v in fields.items():
            patch.append({
                "op": "add",
                "path": f"/fields/{k}",
                "value": v
            })
        hdrs = {**self.headers, "Content-Type": "application/json-patch+json"}
        try:
            resp = requests.patch(url, auth=self.auth, headers=hdrs, json=patch, timeout=10)
            resp.raise_for_status()
        except Exception as ex:
            print(f"[ADOClient] Work item creation failed: {ex}")
            raise
        data = resp.json()
        return {
            "id": data.get("id"),
            "title": data.get("fields", {}).get("System.Title"),
            "url": data.get("_links", {}).get("html", {}).get("href")
        }

# Example test (remove in prod)
if __name__ == "__main__":
    ORG = os.environ.get("ADO_ORGANIZATION", "your_org")
    PROJ = os.environ.get("ADO_PROJECT", "your_proj")
    PAT = os.environ.get("ADO_PAT", "your_pat")
    client = ADOClient(ORG, PROJ, PAT)
    results = client.search_stories("filter button not working")
    from pprint import pprint
    pprint(results)
