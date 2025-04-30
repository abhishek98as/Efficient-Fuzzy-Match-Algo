import torch
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import warnings
import time
import concurrent.futures
from tqdm import tqdm
import os
import traceback
import sys
import psutil
import math
from datetime import datetime, timedelta
import gc
warnings.filterwarnings('ignore')

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SystemMonitor:
    """Monitors system resources (CPU, RAM)"""
    @staticmethod
    def get_system_info():
        """Get system information"""
        cores_physical = psutil.cpu_count(logical=False)
        cores_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        max_freq = cpu_freq.max if cpu_freq else "Unknown"
        
        ram = psutil.virtual_memory()
        total_ram = ram.total / (1024 * 1024)  # MB
        available_ram = ram.available / (1024 * 1024)  # MB
        
        gpu_info = "Integrated graphics"
        if torch.cuda.is_available():
            gpu_info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} device(s))"
            
        return {
            "System": f"{os.name} {platform.system()} {platform.release()}",
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "CPU Cores": f"{cores_physical} physical, {cores_logical} logical",
            "CPU Max Frequency": f"{max_freq} MHz" if max_freq != "Unknown" else max_freq,
            "Total RAM": f"{total_ram:.0f} MB",
            "Available RAM": f"{available_ram:.0f} MB",
            "GPU": gpu_info
        }
    
    @staticmethod
    def get_resource_usage():
        """Get current resource usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_used = ram.used / (1024 * 1024)  # MB
        
        gpu_memory = 0
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            except:
                pass
                
        return {
            "CPU Usage": f"{cpu_percent:.1f}%",
            "RAM Usage": f"{ram_percent:.1f}% ({ram_used:.0f} MB)",
            "GPU Memory": f"{gpu_memory:.0f} MB" if gpu_memory > 0 else "N/A"
        }

    @staticmethod
    def get_optimal_workers(reserve_cores=1):
        """Determine optimal number of worker threads based on system resources"""
        logical_cores = psutil.cpu_count(logical=True)
        available_cores = max(1, logical_cores - reserve_cores)
        
        # Also check RAM constraints
        ram = psutil.virtual_memory()
        available_ram_gb = ram.available / (1024 * 1024 * 1024)
        
        # Heuristic: Allocate ~1 worker per 2GB available RAM, unless CPU constrained
        ram_based_workers = max(1, int(available_ram_gb / 2))
        
        # Take the minimum to avoid over-allocation
        optimal_workers = min(available_cores, ram_based_workers)
        
        return optimal_workers

class TextMatchingSystem:
    def __init__(self, use_gpu=True, batch_size=128, cache_dir=None):
        # Check for GPU
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Get system information
        try:
            import platform
            system_info = SystemMonitor.get_system_info()
            print("System Information:")
            for key, value in system_info.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Could not retrieve system information: {e}")
        
        # Initialize models
        self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(self.device)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.batch_size = batch_size
        
        # Cache directory for embeddings
        self.cache_dir = cache_dir
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Preprocess text with normalization, stemming, and stopword removal"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Simple word split instead of word_tokenize to avoid potential NLTK issues
        tokens = text.split()
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def compute_fuzzy_similarity(self, text1, text2):
        """Compute fuzzy matching score between two texts"""
        if pd.isna(text1) or pd.isna(text2):
            return 0.0
        return fuzz.token_sort_ratio(str(text1).lower(), str(text2).lower()) / 100.0
    
    def compute_batch_fuzzy_similarity(self, texts, target_texts, max_workers=None):
        """Compute fuzzy matching scores in parallel"""
        results = []
        total_comparisons = len(texts) * len(target_texts)
        
        # Use optimal number of workers if not specified
        if max_workers is None:
            max_workers = SystemMonitor.get_optimal_workers()
            
        print(f"Using {max_workers} threads for fuzzy similarity computation")
        
        # Create a progress bar
        progress_bar = tqdm(total=total_comparisons, desc="Fuzzy Matching")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create batches for better progress tracking
            batch_size = 1000  # Process comparisons in batches
            batches = []
            
            for i in range(0, len(texts), batch_size):
                for j in range(0, len(target_texts), batch_size):
                    text_batch = texts[i:i+batch_size]
                    target_batch = target_texts[j:j+batch_size]
                    batches.append((text_batch, target_batch, i, j))
            
            # Process batches
            for text_batch, target_batch, base_i, base_j in batches:
                futures = []
                
                for i, text in enumerate(text_batch):
                    for j, target in enumerate(target_batch):
                        futures.append(executor.submit(
                            self.compute_fuzzy_similarity, text, target
                        ))
                
                for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                    batch_i = idx // len(target_batch)
                    batch_j = idx % len(target_batch)
                    i = base_i + batch_i
                    j = base_j + batch_j
                    
                    try:
                        score = future.result()
                        results.append((i, j, score))
                    except Exception as e:
                        print(f"Error processing fuzzy match: {e}")
                        results.append((i, j, 0.0))
                    
                    progress_bar.update(1)
        
        progress_bar.close()
        return results
    
    def compute_semantic_embeddings(self, texts, cache_key=None):
        """Compute semantic embeddings for a list of texts"""
        # Try to load from cache first
        if cache_key and self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
            if os.path.exists(cache_path):
                try:
                    print(f"Loading embeddings from cache: {cache_path}")
                    return torch.load(cache_path, map_location=self.device)
                except Exception as e:
                    print(f"Error loading cache: {e}")
        
        embeddings = []
        
        # Process in batches with progress bar
        total_batches = math.ceil(len(texts) / self.batch_size)
        progress_bar = tqdm(total=total_batches, desc="Computing Embeddings")
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_embeddings = self.sentence_transformer.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            progress_bar.update(1)
            
        progress_bar.close()
            
        # Concatenate all batch embeddings
        if embeddings:
            all_embeddings = torch.cat(embeddings, dim=0)
            
            # Save to cache if needed
            if cache_key and self.cache_dir:
                cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
                try:
                    torch.save(all_embeddings, cache_path)
                    print(f"Saved embeddings to cache: {cache_path}")
                except Exception as e:
                    print(f"Error saving cache: {e}")
                    
            return all_embeddings
        return None
    
    def compute_semantic_similarity_matrix(self, embeddings1, embeddings2):
        """Compute semantic similarity matrix between two embedding sets"""
        # Process in batches to avoid memory issues
        similarity_matrix = torch.zeros((embeddings1.shape[0], embeddings2.shape[0]))
        
        total_batches1 = math.ceil(embeddings1.shape[0] / self.batch_size)
        total_batches2 = math.ceil(embeddings2.shape[0] / self.batch_size)
        total_iterations = total_batches1 * total_batches2
        
        progress_bar = tqdm(total=total_iterations, desc="Computing Similarity Matrix")
        
        for i in range(0, embeddings1.shape[0], self.batch_size):
            batch1 = embeddings1[i:i+self.batch_size]
            
            for j in range(0, embeddings2.shape[0], self.batch_size):
                batch2 = embeddings2[j:j+self.batch_size]
                
                # Compute cosine similarity for the current batch
                batch_sim = util.cos_sim(batch1, batch2)
                
                # Store in the result matrix
                similarity_matrix[i:i+batch1.shape[0], j:j+batch2.shape[0]] = batch_sim
                
                progress_bar.update(1)
                
                # Explicitly clear GPU cache periodically if using CUDA
                if self.device == 'cuda' and (i+j) % (self.batch_size*10) == 0:
                    torch.cuda.empty_cache()
                    
        progress_bar.close()
        
        return similarity_matrix.cpu().numpy()
    
    def compute_tfidf_similarity(self, texts, target):
        """Compute TF-IDF vector similarity"""
        # Fit vectorizer on all texts including target
        all_texts = texts + [target]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # Split matrix
        text_tfidf = tfidf_matrix[:-1]
        target_tfidf = tfidf_matrix[-1].reshape(1, -1)
        
        # Compute cosine similarity
        similarities = cosine_similarity(text_tfidf, target_tfidf).flatten()
        
        return similarities
    
    def process_excel_duplicates(self, excel_path, name_column='Employer Name', id_column='ID', 
                                 threshold=0.85, max_workers=None, weights=None, chunk_size=None):
        """
        Process Excel file to find duplicate company names using multithreading
        
        Parameters:
        - excel_path: Path to Excel file
        - name_column: Column containing company names
        - id_column: Column containing IDs
        - threshold: Similarity threshold to consider as duplicate
        - max_workers: Maximum number of threads to use
        - weights: Dictionary with weights for each method
        - chunk_size: Size of chunks to process at once
        
        Returns:
        - DataFrame with duplicate groups
        """
        start_time = total_time = time.time()
        
        # Auto-determine optimal parameters if not provided
        if max_workers is None:
            max_workers = SystemMonitor.get_optimal_workers()
            print(f"Auto-selected {max_workers} worker threads based on system resources")
        
        # Determine chunk size based on available RAM
        if chunk_size is None:
            ram = psutil.virtual_memory()
            available_ram_gb = ram.available / (1024 * 1024 * 1024)
            chunk_size = min(5000, max(500, int(available_ram_gb * 500)))  # ~2MB per name embedding
            print(f"Auto-selected chunk size of {chunk_size} based on available RAM ({available_ram_gb:.1f} GB)")
        
        # Default weights
        if weights is None:
            weights = {'semantic': 0.6, 'fuzzy': 0.4}
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        print(f"Using weights: {weights}")
        print(f"Using similarity threshold: {threshold}")
        
        print(f"Reading Excel file: {excel_path}")
        # Read Excel file
        df = pd.read_excel(excel_path)
        print(f"Loaded {len(df)} records in {time.time() - start_time:.2f} seconds")
        
        # Check for missing company names
        missing_names = df[pd.isna(df[name_column])].shape[0]
        if missing_names > 0:
            print(f"Warning: Found {missing_names} records with missing company names!")
        
        # Clean data
        start_time = time.time()
        print("Preprocessing company names...")
        df['Clean_Name'] = df[name_column].apply(self.preprocess_text)
        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        
        # Create unique company list
        unique_companies = df[[name_column, 'Clean_Name']].drop_duplicates(subset='Clean_Name').reset_index(drop=True)
        unique_names = unique_companies[name_column].tolist()
        clean_unique_names = unique_companies['Clean_Name'].tolist()
        
        print(f"Found {len(unique_companies)} unique company names after cleaning")
        
        # Initialize result dataframe with all companies
        result_df = df[[name_column, id_column]].copy()
        result_df['Duplicate_Group'] = None
        result_df['Match_Percentage'] = 0.0
        
        # Break processing into chunks to avoid memory issues
        chunk_count = (len(unique_names) + chunk_size - 1) // chunk_size
        
        if chunk_count > 1:
            print(f"Processing in {chunk_count} chunks of {chunk_size} companies each")
        
        group_id = 0
        all_groups = []
        match_percentages = {}
        
        # Create cache directory
        cache_dir = os.path.join(os.path.dirname(excel_path), "embedding_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Overall progress tracking
        overall_start = time.time()
        chunks_processed = 0
        
        for chunk_idx in range(chunk_count):
            chunk_start = time.time()
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(unique_names))
            
            print(f"\nProcessing chunk {chunk_idx+1}/{chunk_count} ({start_idx+1}-{end_idx} of {len(unique_names)} companies)")
            
            chunk_names = unique_names[start_idx:end_idx]
            chunk_clean_names = clean_unique_names[start_idx:end_idx]
            
            # Create a cache key based on the first and last name plus count
            cache_key = f"chunk_{hash(''.join(chunk_clean_names[:5] + chunk_clean_names[-5:]))}"
            
            # Compute semantic embeddings
            start_time = time.time()
            print(f"Computing embeddings for chunk {chunk_idx+1}/{chunk_count}...")
            embeddings = self.compute_semantic_embeddings(chunk_clean_names, cache_key=cache_key)
            embedding_time = time.time() - start_time
            print(f"Embeddings computed in {embedding_time:.2f} seconds")
            
            # Resource usage checkpoint
            resources = SystemMonitor.get_resource_usage()
            print(f"Resource usage - {resources['CPU Usage']} CPU, {resources['RAM Usage']} RAM, {resources['GPU Memory']} GPU")
            
            # Compute similarity matrix
            start_time = time.time()
            print("Computing similarity matrix...")
            semantic_matrix = self.compute_semantic_similarity_matrix(embeddings, embeddings)
            similarity_time = time.time() - start_time
            print(f"Similarity matrix computed in {similarity_time:.2f} seconds")
            
            # Compute fuzzy similarities in parallel
            start_time = time.time()
            print("Computing fuzzy similarities...")
            fuzzy_results = self.compute_batch_fuzzy_similarity(
                chunk_clean_names, chunk_clean_names, max_workers=max_workers
            )
            
            # Convert to matrix
            fuzzy_matrix = np.zeros((len(chunk_names), len(chunk_names)))
            for i, j, score in fuzzy_results:
                fuzzy_matrix[i, j] = score
            
            fuzzy_time = time.time() - start_time
            print(f"Fuzzy similarities computed in {fuzzy_time:.2f} seconds")
            
            # Compute hybrid similarity
            hybrid_matrix = (
                weights.get('semantic', 0) * semantic_matrix + 
                weights.get('fuzzy', 0) * fuzzy_matrix
            )
            
            # Find duplicate groups
            start_time = time.time()
            print("Finding duplicate groups...")
            processed = set()
            chunk_groups = []
            
            progress_bar = tqdm(total=len(chunk_names), desc="Finding Groups")
            
            for i in range(len(chunk_names)):
                if i in processed:
                    progress_bar.update(1)
                    continue
                    
                # Find all companies similar to this one
                similar_indices = [j for j in range(len(chunk_names)) 
                                 if hybrid_matrix[i, j] >= threshold and i != j]
                
                if similar_indices:
                    # Create a group with the current company and all similar ones
                    group = [(start_idx + i, chunk_names[i], 100)]  # Base company is 100% match to itself
                    
                    # Add similar companies with match percentages
                    for j in similar_indices:
                        match_percent = round(hybrid_matrix[i, j] * 100, 1)
                        group.append((start_idx + j, chunk_names[j], match_percent))
                        # Store match percentages for later use
                        match_key = (group_id + 1, chunk_names[j])
                        match_percentages[match_key] = match_percent
                    
                    # Mark all as processed
                    processed.add(i)
                    processed.update(similar_indices)
                    
                    chunk_groups.append(group)
                    group_id += 1
                
                progress_bar.update(1)
            
            progress_bar.close()
            
            all_groups.extend(chunk_groups)
            print(f"Found {len(chunk_groups)} duplicate groups in chunk {chunk_idx+1}")
            
            # Calculate and show progress info
            chunks_processed += 1
            elapsed = time.time() - overall_start
            avg_chunk_time = elapsed / chunks_processed
            remaining_chunks = chunk_count - chunks_processed
            eta = avg_chunk_time * remaining_chunks
            
            print(f"\nChunk {chunk_idx+1} processed in {time.time() - chunk_start:.2f} seconds")
            print(f"Overall progress: {chunks_processed}/{chunk_count} chunks ({chunks_processed/chunk_count*100:.1f}%)")
            if remaining_chunks > 0:
                eta_str = str(timedelta(seconds=int(eta)))
                print(f"Estimated time remaining: {eta_str} (ETA: {(datetime.now() + timedelta(seconds=eta)).strftime('%H:%M:%S')})")
            
            # Force garbage collection
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Update result dataframe with group IDs and match percentages
        print("Updating result dataframe with duplicate groups...")
        processed_ids = set()
        
        for g_id, group in enumerate(all_groups):
            group_indices = []
            for global_idx, company_name, match_percent in group:
                # Find all rows with this company name
                company_rows = df[df[name_column] == company_name].index.tolist()
                group_indices.extend(company_rows)
                
                # Update match percentage for these rows
                match_key = (g_id + 1, company_name)
                if match_key in match_percentages:
                    result_df.loc[company_rows, 'Match_Percentage'] = match_percentages[match_key]
                
            # Update duplicate group for all found indices
            if group_indices:
                result_df.loc[group_indices, 'Duplicate_Group'] = g_id + 1
                processed_ids.update(group_indices)
        
        # Set ungrouped companies to group 0
        ungrouped_indices = set(range(len(df))) - processed_ids
        if ungrouped_indices:
            result_df.loc[list(ungrouped_indices), 'Duplicate_Group'] = 0
        
        total_elapsed = time.time() - total_time
        print(f"\nTotal processing time: {total_elapsed:.2f} seconds ({timedelta(seconds=int(total_elapsed))})")
        print(f"Found {len(all_groups)} duplicate groups among {len(df)} records")
        
        # Print performance metrics
        records_per_second = len(df) / total_elapsed
        print(f"Performance: {records_per_second:.1f} records/second")
        
        # Final resource usage
        resources = SystemMonitor.get_resource_usage()
        print(f"Final resource usage - {resources['CPU Usage']} CPU, {resources['RAM Usage']} RAM, {resources['GPU Memory']} GPU")
        
        return result_df
    
    def style_dataframe(self, df):
        """Apply color styling to scores in DataFrame"""
        # Map score to color
        def get_color(score):
            if score >= 0.9:
                return 'background-color: green; color: white'
            elif 0.8 <= score < 0.9:
                return 'background-color: orange; color: black'
            else:
                return 'background-color: red; color: white'
        
        # Apply styling to all score columns
        score_columns = [col for col in df.columns if 'Score' in col or 'Percentage' in col]
        df_styled = df.style.applymap(get_color, subset=score_columns)
        
        return df_styled
    
    def export_duplicate_groups(self, df, output_path, 
                               name_column='Employer Name', id_column='ID', 
                               group_column='Duplicate_Group', percent_column='Match_Percentage'):
        """
        Export duplicate groups to separate Excel sheets
        
        Parameters:
        - df: DataFrame with duplicate groups
        - output_path: Path to save Excel file
        - name_column: Column containing company names
        - id_column: Column containing IDs
        - group_column: Column containing group IDs
        - percent_column: Column containing match percentages
        """
        print(f"Exporting duplicate groups to {output_path}...")
        
        # Create a summary of duplicate groups
        summary_data = []
        group_ids = sorted(df[df[group_column] > 0][group_column].unique())
        
        for group_id in group_ids:
            group_df = df[df[group_column] == group_id]
            primary_company = group_df.iloc[0][name_column]
            companies_count = len(group_df)
            avg_match = group_df[percent_column].mean()
            
            summary_data.append({
                'Group ID': int(group_id),
                'Primary Company': primary_company,
                'Number of Companies': companies_count,
                'Average Match %': avg_match
            })
            
        summary_df = pd.DataFrame(summary_data)
        
        # Create a Pandas Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # First, write a summary sheet
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Then write all records
            df.to_excel(writer, sheet_name='All Records', index=False)
            
            # Create a sheet for each duplicate group
            for group_id in group_ids:
                # Get companies in this group
                group_df = df[df[group_column] == group_id][[name_column, id_column, percent_column]]
                # Sort by match percentage
                group_df = group_df.sort_values(percent_column, ascending=False)
                
                # Write to sheet
                sheet_name = f'Group {int(group_id)}'
                if len(sheet_name) > 31:  # Excel sheet names are limited to 31 chars
                    sheet_name = sheet_name[:31]
                    
                group_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Create a sheet for non-duplicate companies
            non_duplicates = df[df[group_column] == 0][[name_column, id_column]]
            if len(non_duplicates) > 0:
                non_duplicates.to_excel(writer, sheet_name='No Duplicates', index=False)
        
        print(f"Export completed successfully!")

# Example usage
if __name__ == "__main__":
    try:
        import platform
        import psutil
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import platform
        import psutil
    
    print("==== Company Duplicate Finder ====")
    print(f"Starting process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create cache directory
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Initialize matcher with GPU support if available
    matcher = TextMatchingSystem(use_gpu=True, batch_size=128, cache_dir=cache_dir)
    
    # Default path - can be changed
    excel_path = "C:\\Users\\asingh50\\Documents\\Dublicate Sort\\New_Company List.xlsx"
    
    # If file doesn't exist, ask for path
    if not os.path.exists(excel_path):
        print(f"File not found: {excel_path}")
        excel_path = input("Please enter the path to your Excel file: ")
    
    # Detect optimal parameters based on system
    max_workers = SystemMonitor.get_optimal_workers()
    
    # Process Excel file with multithreading
    result_df = matcher.process_excel_duplicates(
        excel_path=excel_path,
        threshold=0.85,  # Similarity threshold
        max_workers=max_workers,  # Auto-detect number of threads
        weights={'semantic': 0.6, 'fuzzy': 0.4},  # Give more weight to semantic matching
        # chunk_size is auto-detected based on available RAM
    )
    
    # Export results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = f"C:\\Users\\asingh50\\Documents\\Dublicate Sort\\Duplicate_Results_{timestamp}.xlsx"
    matcher.export_duplicate_groups(result_df, output_path)
    
    # Show summary
    duplicate_count = len(result_df[result_df['Duplicate_Group'] > 0])
    unique_count = len(result_df[result_df['Duplicate_Group'] == 0])
    group_count = result_df['Duplicate_Group'].max()
    
    print("\n===== SUMMARY =====")
    print(f"Total records: {len(result_df)}")
    print(f"Records in duplicate groups: {duplicate_count} ({duplicate_count/len(result_df)*100:.1f}%)")
    print(f"Records with no duplicates: {unique_count} ({unique_count/len(result_df)*100:.1f}%)")
    print(f"Number of duplicate groups: {int(group_count)}")
    print(f"Results saved to: {output_path}")
    print("===================")