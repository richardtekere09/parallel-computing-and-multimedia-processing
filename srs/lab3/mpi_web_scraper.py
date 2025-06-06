
import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urljoin, urlparse
from mpi4py import MPI
import sys

class WikipediaAnalyzer:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_page_content(self, url, max_retries=3):
        """Fetch page content with retry mechanism"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.text
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)
        return None
    
    def extract_province_links(self, main_page_url):
        """Extract all province Wikipedia links from the main page"""
        print(f"Fetching main page: {main_page_url}")
        content = self.get_page_content(main_page_url)
        if not content:
            return []
        
        soup = BeautifulSoup(content, 'html.parser')
        province_links = []
        
        # Find all tables that might contain province information
        tables = soup.find_all('table', class_='wikitable')
        
        for table in tables:
            # Look for links in table cells
            links = table.find_all('a', href=True)
            for link in links:
                href = link.get('href', '')
                if href.startswith('/wiki/') and not any(x in href for x in [':', '#', 'File:', 'Category:']):
                    # Check if this looks like a province page
                    title = link.get('title', '').lower()
                    text = link.get_text().lower()
                    
                    # Filter for Indonesian provinces
                    if any(keyword in title or keyword in text for keyword in 
                          ['province', 'regency', 'city', 'jakarta', 'yogyakarta', 
                           'aceh', 'bali', 'java', 'sumatra', 'kalimantan', 'sulawesi', 'papua']):
                        full_url = urljoin(self.base_url, href)
                        if full_url not in province_links:
                            province_links.append(full_url)
        
        # Also look for direct province links in the page content
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', '')
            if href.startswith('/wiki/') and '_Province' in href:
                full_url = urljoin(self.base_url, href)
                if full_url not in province_links:
                    province_links.append(full_url)
        
        print(f"Found {len(province_links)} potential province links")
        return province_links[:38]  # Limit to actual number of Indonesian provinces
    
    def analyze_page_sections(self, url):
        """Analyze sections of a Wikipedia page"""
        print(f"Analyzing: {url}")
        content = self.get_page_content(url)
        if not content:
            return None
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract page title
        title_tag = soup.find('h1', class_='firstHeading')
        page_title = title_tag.get_text() if title_tag else urlparse(url).path.split('/')[-1]
        
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        # Find all content elements
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            return None
        
        # Process all elements in the main content
        for element in content_div.find_all(['h2', 'h3', 'p', 'table', 'img', 'sup']):
            if element.name in ['h2', 'h3']:
                # Process previous section if it has content
                if current_content:
                    sections[current_section] = self.analyze_section_content(current_content)
                
                # Start new section
                span = element.find('span', class_='mw-headline')
                if span:
                    current_section = span.get_text().strip()
                    current_content = []
            else:
                current_content.append(element)
        
        # Process last section
        if current_content:
            sections[current_section] = self.analyze_section_content(current_content)
        
        # Find best section
        best_section = max(sections.keys(), key=lambda k: sections[k]['score']) if sections else "None"
        
        return {
            'province': page_title,
            'url': url,
            'sections': sections,
            'best_section': best_section
        }
    
    def analyze_section_content(self, elements):
        """Analyze content elements and count I, T, R"""
        images = 0
        tables = 0
        references = 0
        
        for element in elements:
            if element.name == 'img':
                images += 1
            elif element.name == 'table':
                tables += 1
            elif element.name == 'sup':
                # Check if it's a reference
                if 'reference' in element.get('class', []) or element.find('a'):
                    references += 1
            else:
                # Check for nested elements
                images += len(element.find_all('img'))
                tables += len(element.find_all('table'))
                # Count reference superscripts
                sup_refs = element.find_all('sup', class_='reference')
                references += len(sup_refs)
                # Also count citation links
                cite_links = element.find_all('a', href=lambda x: x and '#cite' in x)
                references += len(cite_links)
        
        score = images + tables + references
        return {
            'images': images,
            'tables': tables,
            'references': references,
            'score': score
        }

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()
    
    analyzer = WikipediaAnalyzer()
    main_url = "https://en.wikipedia.org/wiki/Provinces_of_Indonesia"
    
    if rank == 0:
        print(f"Starting analysis with {size} MPI processes")
        
        # Extract province links
        province_links = analyzer.extract_province_links(main_url)
        print(f"Total provinces to analyze: {len(province_links)}")
        
        # Distribute links among processes
        links_per_process = len(province_links) // size
        remainder = len(province_links) % size
        
        # Create chunks for each process
        chunks = []
        start_idx = 0
        for i in range(size):
            chunk_size = links_per_process + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            chunks.append(province_links[start_idx:end_idx])
            start_idx = end_idx
        
        print(f"Distribution: {[len(chunk) for chunk in chunks]}")
    else:
        chunks = None
    
    # Scatter the work
    my_links = comm.scatter(chunks, root=0)
    
    print(f"Process {rank}: Received {len(my_links)} links to process")
    
    # Process assigned links
    my_results = []
    for i, link in enumerate(my_links):
        print(f"Process {rank}: Analyzing {i+1}/{len(my_links)} - {link}")
        result = analyzer.analyze_page_sections(link)
        if result:
            my_results.append(result)
        time.sleep(0.5)  # Be nice to Wikipedia servers
    
    # Gather results back to rank 0
    all_results = comm.gather(my_results, root=0)
    
    if rank == 0:
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Flatten results
        final_results = []
        for process_results in all_results:
            final_results.extend(process_results)
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE - {len(final_results)} provinces analyzed")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"MPI processes: {size}")
        print(f"{'='*80}")
        
        # Display results
        for result in final_results:
            print(f"\nProvince: {result['province']}")
            print(f"Best Section: {result['best_section']} — Score: {result['sections'][result['best_section']]['score']}")
            
            # Show all sections
            for section_name, section_data in result['sections'].items():
                marker = ">>> " if section_name == result['best_section'] else "    "
                print(f"{marker}{section_name}: I={section_data['images']}, "
                      f"T={section_data['tables']}, R={section_data['references']}, "
                      f"Score={section_data['score']}")
        
        # Performance summary
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUMMARY")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Average time per province: {execution_time/len(final_results):.2f} seconds")
        print(f"Processes used: {size}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()