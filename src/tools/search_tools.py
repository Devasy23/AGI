from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from .base_tool import BaseTool
import requests

class BrowserSearchTool(BaseTool):
    name = "tool_browser"
    description = "Search on DuckDuckGo browser by passing the input `query`"
    
    def __init__(self):
        self._search = DuckDuckGoSearchRun()
    
    def invoke(self, query: str) -> str:
        return self._search.run(query)

class WikipediaSearchTool(BaseTool):
    name = "tool_wikipedia"
    description = "Search on Wikipedia by passing the input `query`. The input `query` must be short keywords, not a long text"
    
    def __init__(self):
        self._search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    def invoke(self, query: str) -> str:
        return self._search.run(query)

class FinalAnswerTool(BaseTool):
    name = "final_answer"
    description = "Returns a natural language response to the user by passing the input `text`. You should provide as much context as possible and specify the source of the information."
    
    def invoke(self, text: str) -> str:
        return text

class GitHubIssuesTool(BaseTool):
    name = "tool_github_issues"
    description = "Scan GitHub issues from specified repositories and return the top 5 issues compatible with the user's skills."
    
    def __init__(self):
        self.api_url = "https://api.github.com/repos/{owner}/{repo}/issues"
    
    def invoke(self, repos: list, skills: list) -> list:
        all_issues = []
        for repo in repos:
            page = 1
            while True:
                issues = self.fetch_issues(repo, page)
                if not issues:
                    break
                all_issues.extend(issues)
                page += 1
        
        filtered_issues = self.filter_issues(all_issues, skills)
        assigned_issues = self.assign_issues_to_agents(filtered_issues, skills)
        top_issues = self.get_top_issues(assigned_issues)
        return top_issues
    
    def fetch_issues(self, repo: str, page: int = 1, per_page: int = 100) -> list:
        owner, repo_name = repo.split('/')
        url = f"{self.api_url.format(owner=owner, repo=repo_name)}?page={page}&per_page={per_page}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch issues from {repo}: {response.status_code}")
            return []

    def filter_issues(self, issues: list, skills: list) -> list:
        filtered_issues = []
        for issue in issues:
            if any(skill.lower() in issue['title'].lower() or skill.lower() in issue['body'].lower() for skill in skills):
                filtered_issues.append(issue)
        return filtered_issues

    def assign_issues_to_agents(self, issues: list, skills: list) -> list:
        assigned_issues = []
        for issue in issues:
            compatibility_score = sum(skill.lower() in issue['title'].lower() or skill.lower() in issue['body'].lower() for skill in skills)
            assigned_issues.append((issue, compatibility_score))
        return assigned_issues

    def get_top_issues(self, assigned_issues: list, top_n: int = 5) -> list:
        assigned_issues.sort(key=lambda x: x[1], reverse=True)
        return [issue for issue, score in assigned_issues[:top_n]]
