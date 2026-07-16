"""GraphQL queries for repository acquisition."""

# Discovers repositories via GraphQL search — no REST needed
SEARCH_REPOSITORIES_QUERY = """
query SearchRepositories($query: String!, $after: String) {
  search(query: $query, type: REPOSITORY, first: 25, after: $after) {
    repositoryCount
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      ... on Repository {
        id
        databaseId
        nameWithOwner
        name
        owner { login }
        stargazerCount
        description
        pushedAt
      }
    }
  }
  rateLimit {
    cost
    remaining
    resetAt
  }
}
"""


def build_batch_metadata_query(repos: list[tuple[str, str]]) -> str:
    """Ultra-lean batch query — metadata + topics + languages only.
    NO readme fields: readme text can be 30KB+ per repo and causes 502 in batches.
    READMEs are fetched separately via individual queries."""
    parts = ["query GetBatchMetadata {"]
    for i, (owner, name) in enumerate(repos):
        alias = f"repo_{i}"
        escaped_owner = owner.replace('\\', '\\\\').replace('"', '\\"')
        escaped_name = name.replace('\\', '\\\\').replace('"', '\\"')
        parts.append(f"""
  {alias}: repository(owner: "{escaped_owner}", name: "{escaped_name}") {{
    id
    databaseId
    nameWithOwner
    name
    description
    url
    homepageUrl
    createdAt
    updatedAt
    pushedAt
    stargazerCount
    forkCount
    owner {{ login __typename }}
    watchers {{ totalCount }}
    issues(states: OPEN) {{ totalCount }}
    pullRequests(states: OPEN) {{ totalCount }}
    repositoryTopics(first: 20) {{
      nodes {{ topic {{ name }} }}
    }}
    languages(first: 10, orderBy: {{field: SIZE, direction: DESC}}) {{
      edges {{ size node {{ name }} }}
    }}
    defaultBranchRef {{
      target {{
        ... on Commit {{
          history(first: 10) {{ nodes {{ committedDate }} }}
        }}
      }}
    }}
  }}""")

    parts.append("""
  rateLimit { cost remaining resetAt }
}""")
    return "\n".join(parts)


# Fetches only the README for one repo — called individually after batch metadata
GET_README_QUERY = """
query GetReadme($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    readme1: object(expression: "HEAD:README.md") { ... on Blob { text } }
    readme2: object(expression: "HEAD:readme.md") { ... on Blob { text } }
    readme3: object(expression: "HEAD:README.rst") { ... on Blob { text } }
    readme4: object(expression: "HEAD:README.txt") { ... on Blob { text } }
    readme5: object(expression: "HEAD:README")     { ... on Blob { text } }
  }
}
"""

GET_REPOSITORY_QUERY = """
query GetRepository($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id
    databaseId
    name
    nameWithOwner
    description
    url
    homepageUrl
    createdAt
    updatedAt
    pushedAt
    stargazerCount
    forkCount
    
    watchers {
      totalCount
    }
    issues(states: OPEN) {
      totalCount
    }
    pullRequests(states: OPEN) {
      totalCount
    }
    
    repositoryTopics(first: 20) {
      nodes {
        topic {
          name
        }
      }
    }
    
    languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
      edges {
        size
        node {
          name
        }
      }
    }
    
    licenseInfo {
      name
      spdxId
    }
    
    owner {
      login
      __typename
    }
    
    defaultBranchRef {
      name
      target {
        ... on Commit {
          history(first: 30) {
            nodes {
              committedDate
            }
          }
        }
      }
    }
    
    readme1: object(expression: "HEAD:README.md") {
      ... on Blob {
        text
      }
    }
    readme2: object(expression: "HEAD:readme.md") {
      ... on Blob {
        text
      }
    }
    readme3: object(expression: "HEAD:README.rst") {
      ... on Blob {
        text
      }
    }
    readme4: object(expression: "HEAD:README.txt") {
      ... on Blob {
        text
      }
    }
    readme5: object(expression: "HEAD:README") {
      ... on Blob {
        text
      }
    }

    stargazers(last: 100) {
      edges {
        starredAt
      }
    }
  }
  
  rateLimit {
    cost
    remaining
    resetAt
  }
}
"""

