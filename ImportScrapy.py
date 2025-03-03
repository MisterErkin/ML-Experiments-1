import scrapy
from scrapy.crawler import CrawlerProcess
import os
import csv
import tkinter as tk
from tkinter import ttk
import threading

class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]
    scraped_books = 0  # Track how many books have been scraped
    max_books = 700  # Default target number of books
    collected_data = []  # Store scraped items in memory before writing to file
    
    rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}  # Convert text rating to numeric
    
    def __init__(self, max_books=700, progress_var=None, progress_bar=None, *args, **kwargs):
        super(BooksSpider, self).__init__(*args, **kwargs)
        self.max_books = int(max_books)
        self.progress_var = progress_var
        self.progress_bar = progress_bar
    
    def parse(self, response):
        if self.scraped_books >= self.max_books:
            return  # Stop scraping when max_books are collected
        
        books = response.css(".product_pod")
        
        total_books = min(self.max_books, 1000)  # Avoid exceeding dataset limits
        progress_step = 100 / total_books  # Calculate step for progress bar

        for book in books:
            if self.scraped_books >= self.max_books:
                break  # Stop if we've reached the limit
            
            title = book.css("h3 a::attr(title)").get().replace(',', ' -')  # Fix multi-title issue
            product_link = response.urljoin(book.css("h3 a::attr(href)").get())
            price = book.css(".price_color::text").get()
            rating_text = book.css("p.star-rating::attr(class)").get().split()[1]
            rating = self.rating_map.get(rating_text, 0)  # Convert rating to number, default to 0 if not found
            
            self.scraped_books += 1  # Increment count
            self.collected_data.append({
                "Title": title,
                "Price": price,
                "Rating": rating,
                "Product Link": product_link
            })
            
            # Update progress bar
            if self.progress_var and self.progress_bar:
                self.progress_var.set(min(100, self.scraped_books * progress_step))
                self.progress_bar.update_idletasks()
        
        next_page = response.css("li.next a::attr(href)").get()
        if next_page and self.scraped_books < self.max_books:
            yield response.follow(next_page, self.parse)
    
    def close(self, reason):
        save_path = os.path.join(os.getcwd(), "books_data.csv")
        
        # Ensure old file is removed before creating a new one
        if os.path.exists(save_path):
            os.remove(save_path)
        
        with open(save_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=';')  # Use semicolon as separator
            writer.writerow(["Title", "Price", "Rating", "Product Link"])  # Write headers
            
            for item in self.collected_data:
                writer.writerow([item["Title"], item["Price"], item["Rating"], item["Product Link"]])
            
        print("Scraping finished. Check 'books_data.csv' in the same directory.")

# Function to display UI for user input
def get_scrape_limit():
    root = tk.Tk()
    root.title("Scraper Settings")
    root.geometry("300x150")
    
    def start_scraping():
        global max_books
        max_books = int(entry.get())
        root.destroy()
    
    tk.Label(root, text="Enter the number of books to scrape:").pack(pady=10)
    entry = tk.Entry(root)
    entry.pack()
    entry.insert(0, "700")
    
    tk.Button(root, text="Start", command=start_scraping).pack(pady=10)
    root.mainloop()
    
    return max_books if 'max_books' in globals() else 700

# Function to run the scraper with UI feedback
def run_scraper():
    max_books = get_scrape_limit()
    
    root = tk.Tk()
    root.title("Scraping in Progress")
    root.geometry("300x150")
    label = tk.Label(root, text="Scraping data, please wait...")
    label.pack(pady=10)
    
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=250, mode="determinate", variable=progress_var)
    progress_bar.pack(pady=10)
    root.update()
    
    def start_scraping():
        process = CrawlerProcess()
        process.crawl(BooksSpider, max_books=max_books, progress_var=progress_var, progress_bar=progress_bar)
        process.start()
        root.destroy()
        print("Scraper execution completed.")
    
    # Run Scrapy in a separate thread to keep the UI responsive
    threading.Thread(target=start_scraping).start()
    root.mainloop()
