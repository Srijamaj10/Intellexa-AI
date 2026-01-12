#step1: Access arXiv using url
import requests 

def search_arxiv_papers(topic:str, max_results: int = 5) -> dict:
    query= "+".join(topic.lower().split())
    for char in list('()" '):
        if char in query:
            print(f"Invalid charcter '{char}' in query: {query}")
            raise ValueError(f"Cannot have character: '{char}' in query: {query}")
    url= (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{query}"
        f"&max_results={max_results}"
        "&sortBy=submittedDate"
        "&sortOrder=descending"
    ) 

    print(f"Making request to arXiv API: {url}")
    resp= requests.get(url)

    if not resp.ok:
        print(f"ArXiv API request failed:{resp.status_code} - {resp.text}")
        raise ValueError(f"Bad response from arXiv API: {resp}\n{resp.text}")
    data = parse_arxiv_xml(resp.text)
    return data


#step2: Parse XML

import xml.etree.ElementTree as ET
def parse_arxiv_xml(xml_content: str) -> dict:
    """Parse the XML content from arXiv API response."""
    entries = []
    ns={
        "atom":"http://www.w3.org/2005/Atom",
        "arxiv":"http://arxiv.org/schemas/atom"
    }
    root = ET.fromstring(xml_content)
    #Loop through each <entry> in the Atom namespace
    for entry in root.findall("atom:entry", ns):
        ##Extract authors
        authors=[
            author.find("atom:name", ns).text
            for author in entry.findall("atom:author", ns)
        ]

        ##Extract categories (term attribute)
        categories=[
            cat.attrib.get("term")
            for cat in entry.findall("atom:category", ns)
        ]

        #Extract PDF link (rel="related" and type="application/pdf")
        pdf_link= None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("type")=="application/pdf":
                pdf_link= link.attrib.get("href")
                break

        entries.append({
            "title": entry.find("atom:title",ns).text,
            "summary": entry.find("atom:summary",ns).text.strip(),
            "authors":authors,
            "categories": categories,
            "pdf":pdf_link
        })

    return {"entries":entries}


#step3: Convert the functionality into a tool
from langchain_core.tools import tool

@tool
def arxiv_search(topic: str) -> list[dict]:
    """Search for recently uploaded arXiv papers

    Args:
        topic: The topic to search for papers about
    
    Returns:
        List of papers with their meatdata including title, authors, summary, etc.
    """
    print("ARXIV Agent called")
    print(f"Searching arXiv for papers about:{topic}")
    papers = search_arxiv_papers(topic)
    if len(papers)==0:
        print(f"No papers found for topic: {topic}")
        raise ValueError(f"No papers found for topic: {topic}")
    print(f"found {len(papers['entries'])} papers about {topic}")
    return papers