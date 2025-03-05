import arxiv
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import logging
import json
import re

class ArxivCrawler:
    def __init__(self, max_papers=100000):
        self.max_papers = max_papers
        self.papers_data = []
        self.index = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=f'arxiv_crawler_{datetime.now().strftime("%Y%m%d")}.log'
        )
        
    def create_search_query(self, start_date, end_date):
        """Create search query for NLP-related papers within a date range"""
        date_query = f"submittedDate:[{start_date} TO {end_date}]"
        category_query = 'cat:cs.CL OR cat:cs.AI OR cat:cs.LG'
        return f"{category_query} AND {date_query}"
    
    def clean_text(self, text):
        """Clean and format text fields"""
        if not text:
            return ""
        # Remove extra whitespace and normalize newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def format_authors(self, authors):
        """Format author names consistently"""
        formatted_authors = []
        for author in authors:
            name = author.name.strip()
            # Remove extra spaces and normalize format
            name = re.sub(r'\s+', ' ', name)
            formatted_authors.append(name)
        return formatted_authors
    
    def parse_paper(self, paper):
        """Extract and format paper information"""
        # Extract arxiv ID from the entry_id URL
        arxiv_id = paper.entry_id.split('/')[-1]
        
        return {
            'arxiv_id': arxiv_id,
            'title': self.clean_text(paper.title),
            'authors': self.format_authors(paper.authors),
            'abstract': self.clean_text(paper.summary),
            'categories': sorted(list(paper.categories)),  # Sort for consistency
            'primary_category': paper.primary_category,
            'published_date': paper.published.strftime('%Y-%m-%d'),
            'updated_date': paper.updated.strftime('%Y-%m-%d'),
            'doi': paper.doi if paper.doi else None,
            'journal_ref': paper.journal_ref if hasattr(paper, 'journal_ref') else None,
            'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            'web_url': f"https://arxiv.org/abs/{arxiv_id}",
            'crawled_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def crawl(self):
        """Crawl arXiv papers using date-based pagination"""
        client = arxiv.Client(
            page_size=100,
            delay_seconds=2,
            num_retries=5
        )
        
        # Start from today and move backwards
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        pbar = tqdm(total=self.max_papers, desc="Crawling papers")
        
        while len(self.papers_data) < self.max_papers:
            try:
                start_str = start_date.strftime('%Y%m%d')
                end_str = end_date.strftime('%Y%m%d')
                
                search = arxiv.Search(
                    query=self.create_search_query(start_str, end_str),
                    max_results=2000,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                batch_papers = []
                earliest_date = None
                
                for paper in client.results(search):
                    try:
                        paper_data = self.parse_paper(paper)
                        batch_papers.append(paper_data)
                        # Track earliest date in this batch
                        paper_date = datetime.strptime(paper_data['published_date'], '%Y-%m-%d')
                        if earliest_date is None or paper_date < earliest_date:
                            earliest_date = paper_date
                    except Exception as e:
                        logging.error(f"Error processing paper {paper.entry_id}: {str(e)}")
                        continue
                
                # Update progress and log batch information
                new_papers = len(batch_papers)
                if new_papers == 0:
                    logging.info(f"No papers found between {start_str} and {end_str}")
                    start_date = start_date - timedelta(days=7)
                else:
                    self.papers_data.extend(batch_papers)
                    pbar.update(new_papers)
                    logging.info(
                        f"Found {new_papers} papers between {start_str} and {end_str}. "
                        f"Earliest paper date: {earliest_date.strftime('%Y-%m-%d') if earliest_date else 'N/A'}"
                    )
                    print(
                        f"\nBatch complete: {new_papers} papers, "
                        f"date range: {earliest_date.strftime('%Y-%m-%d')} to {end_str}"
                    )
                    # Move the date window backwards
                    end_date = start_date
                    start_date = end_date - timedelta(days=7)
                
                # Save intermediate results every 5,000 papers
                if len(self.papers_data) % 5000 < 100:
                    self.save_results(is_intermediate=True)
                
            except Exception as e:
                logging.error(f"Error during crawling: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying
                continue
            
            if len(self.papers_data) >= self.max_papers:
                break
        
        pbar.close()
        self.save_results(is_intermediate=False)
    
    def save_results(self, is_intermediate=False):
        """Save crawled papers to CSV and JSON with proper formatting"""

        # Convert to DataFrame
        df = pd.DataFrame(self.papers_data)
        
        # Sort by published date (newest first)
        df = df.sort_values('published_date', ascending=False)
        
        # Save as CSV
        csv_filename = f'arxiv_papers_{self.index}.csv'
        df.to_csv(csv_filename, index=False, quoting=2)  # Quote all non-numeric fields
        
        # Save as JSON with proper formatting
        json_filename = f'arxiv_papers_{self.index}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(
                df.to_dict('records'),
                f,
                ensure_ascii=False,
                indent=2
            )
        self.index += 1 
        
        logging.info(f"Saved {len(self.papers_data)} papers to {csv_filename} and {json_filename}")

def main():
    crawler = ArxivCrawler(max_papers=100000)
    crawler.crawl()

if __name__ == "__main__":
    main()