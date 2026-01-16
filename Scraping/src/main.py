import logging
import json
import time
import threading
import concurrent.futures

from lib.source_identifier import ArxivHarvester
from lib.source_downloader import SourceDownloader
from lib.reference_extractor import ReferenceExtractor
from lib.statistics_tracker import StatisticsTracker
from lib.paper_organizer import PaperOrganizer
from lib.metadata_updater import MetadataUpdater
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def memory_monitor(stats, stop_event):
    """Background thread to monitor memory usage."""
    while not stop_event.is_set():
        stats.sample_memory()
        time.sleep(5)


def download_paper_sources(paper, downloader, organizer, stats):
    """
    Download all versions of a single paper.
    ----------
    Params
        paper: Paper metadata dictionary
        downloader: SourceDownloader instance
        organizer: PaperOrganizer instance
        stats: StatisticsTracker instance

    Returns:
        Tuple of (arxiv_id, success, processing_time)
    """
    arxiv_id = paper['arxiv_id']
    versions = paper['versions']

    start_time = time.time()

    try:
        logger.info(f"Downloading {arxiv_id} ({len(versions)} versions)")

        # Save metadata
        organizer.save_metadata(arxiv_id, paper)

        # Download and extract all versions
        for version in versions:
            success = downloader.download_and_extract_version(arxiv_id, version)
            if not success:
                logger.warning(f"Warning: Failed to download version {version['version']} of {arxiv_id}")

        processing_time = time.time() - start_time
        logger.info(f"Success: Downloaded {arxiv_id} in {processing_time:.2f}s")

        return arxiv_id, True, processing_time

    except Exception as e:
        logger.error(f"Error: Error downloading {arxiv_id}: {e}")
        import traceback
        traceback.print_exc()

        processing_time = time.time() - start_time
        stats.record_error(f"Download error: {type(e).__name__}")

        return arxiv_id, False, processing_time


def fetch_paper_references(arxiv_id, ref_extractor, organizer, stats):
    """
    Fetch references for a single paper.
    ----------
    Params
        arxiv_id: arXiv ID
        ref_extractor: ReferenceExtractor instance
        organizer: PaperOrganizer instance
        stats: StatisticsTracker instance

    Returns:
        Tuple of (arxiv_id, success, ref_count)
    """
    try:
        logger.info(f"Fetching references for {arxiv_id}")

        ref_data = ref_extractor.fetch_references(arxiv_id)

        if ref_data:
            organizer.save_references(arxiv_id, ref_data)
            ref_count = len(ref_data.get('references', []))
            stats.record_references(ref_count)
            logger.info(f"Success: Found {ref_count} references for {arxiv_id}")
            return arxiv_id, True, ref_count
        else:
            stats.record_references(0)
            logger.warning(f"Error: No references found for {arxiv_id}")
            return arxiv_id, False, 0

    except Exception as e:
        logger.error(f"Error: Error fetching references for {arxiv_id}: {e}")
        stats.record_error(f"Reference error: {type(e).__name__}")
        stats.record_references(0)
        return arxiv_id, False, 0


def main():
    """Main execution pipeline with optimized speed."""
    stats = StatisticsTracker()
    organizer = PaperOrganizer()
    stats.start()

    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(target=memory_monitor, args=(stats, stop_monitoring))
    monitor_thread.start()

    try:
        logger.info("=" * 80)
        logger.info("Starting arXiv Data Scraping")
        logger.info("=" * 80)

        # Step 1: Harvest metadata using arXiv API
        stats.start_phase("Metadata Harvesting (arXiv API)")
        logger.info("\n--- STEP 1: Harvesting metadata via arXiv API ---")

        paper_list_file = config.DATA_DIR / 'paper_list.json'

        if paper_list_file.exists():
            logger.info(f"Loading cached paper list from {paper_list_file}")
            with open(paper_list_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            logger.info(f"Loaded {len(papers)} papers from cache")
        else:
            harvester = ArxivHarvester(
                id_ranges=config.ARXIV_ID_RANGES,
                checkpoint_interval=10  # Save every 10 batches
            )

            # Harvest papers with checkpoint support
            papers = harvester.harvest_all_papers()

            # Record identified papers
            for _ in papers:
                stats.record_paper_identified()

        # Save paper list
        with open(paper_list_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, default=str)

        logger.info(f"Saved {len(papers)} papers to {paper_list_file}")

        stats.end_phase()
        stats.save_checkpoint()

        if not papers:
            logger.warning("No papers found. Exiting.")
            return

        logger.info(f"\n{'='*80}")
        logger.info(f"Found {len(papers)} papers to process")
        logger.info(f"{'='*80}\n")

        # Filter out already complete papers
        papers_to_process = []
        for paper in papers:
            arxiv_id = paper['arxiv_id']
            if organizer.is_paper_complete(arxiv_id):
                logger.info(f"Success: {arxiv_id} already complete, skipping")
                stats.record_paper_success(0)
            else:
                papers_to_process.append(paper)

        logger.info(f"\n{len(papers_to_process)} papers need processing\n")

        if not papers_to_process:
            logger.info("All papers already processed!")
            return

        # Step 2: Download all source papers concurrently
        stats.start_phase("Source Download (Parallel)")
        logger.info("\n--- STEP 2: Downloading all source papers (parallel) ---")
        logger.info(f"Max concurrent downloads: {config.PAPER_DOWNLOAD_BATCH_SIZE}")

        downloader = SourceDownloader(stats_tracker=stats, paper_organizer=organizer)

        download_results = {}
        downloaded_papers = []
        failed_downloads = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.PAPER_DOWNLOAD_BATCH_SIZE) as executor:
            # Submit all download tasks
            future_to_paper = {
                executor.submit(download_paper_sources, paper, downloader, organizer, stats): paper
                for paper in papers_to_process
            }

            completed = 0
            total = len(papers_to_process)

            # Process completed downloads
            for future in concurrent.futures.as_completed(future_to_paper):
                paper = future_to_paper[future]
                completed += 1

                try:
                    arxiv_id, success, processing_time = future.result()
                    download_results[arxiv_id] = success

                    if success:
                        downloaded_papers.append(arxiv_id)
                        stats.record_paper_attempt()
                        logger.info(f"[{completed}/{total}] Success: {arxiv_id} - {processing_time:.2f}s")
                    else:
                        failed_downloads.append(arxiv_id)
                        stats.record_paper_attempt()
                        stats.record_paper_failure()
                        logger.warning(f"[{completed}/{total}] ✗ {arxiv_id} - Failed")

                    # Save checkpoint periodically
                    if completed % config.STATS_SAVE_INTERVAL == 0:
                        stats.save_checkpoint()
                        logger.info(f"--- Checkpoint: {completed}/{total} downloads completed ---")

                except Exception as e:
                    logger.error(f"[{completed}/{total}] ✗ Exception processing {paper['arxiv_id']}: {e}")
                    failed_downloads.append(paper['arxiv_id'])
                    stats.record_paper_attempt()
                    stats.record_paper_failure()

        stats.end_phase()
        stats.save_checkpoint()

        logger.info(f"\n{'='*80}")
        logger.info(f"Download phase completed!")
        logger.info(f"  Successful: {len(downloaded_papers)}/{total}")
        logger.info(f"  Failed: {len(failed_downloads)}/{total}")
        logger.info(f"{'='*80}\n")

        # Step 3: Fetch references for all downloaded papers
        stats.start_phase("Reference Extraction (Parallel)")
        logger.info("\n--- STEP 3: Fetching references for all papers (parallel) ---")
        logger.info(f"Max concurrent API calls: {config.REFERENCE_BATCH_SIZE}")

        ref_extractor = ReferenceExtractor(stats_tracker=stats)

        reference_results = {}
        papers_with_refs = []
        papers_without_refs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.REFERENCE_BATCH_SIZE) as executor:
            # Submit all reference fetching tasks
            future_to_arxiv_id = {
                executor.submit(fetch_paper_references, arxiv_id, ref_extractor, organizer, stats): arxiv_id
                for arxiv_id in downloaded_papers
            }

            completed = 0
            total = len(downloaded_papers)

            # Process completed reference fetches
            for future in concurrent.futures.as_completed(future_to_arxiv_id):
                arxiv_id = future_to_arxiv_id[future]
                completed += 1

                try:
                    arxiv_id_result, success, ref_count = future.result()
                    reference_results[arxiv_id_result] = (success, ref_count)

                    if success:
                        papers_with_refs.append(arxiv_id_result)
                        logger.info(f"[{completed}/{total}] Success: {arxiv_id_result} - {ref_count} refs")
                    else:
                        papers_without_refs.append(arxiv_id_result)
                        logger.warning(f"[{completed}/{total}] ✗ {arxiv_id_result} - No refs")

                    # Save checkpoint periodically
                    if completed % config.STATS_SAVE_INTERVAL == 0:
                        stats.save_checkpoint()
                        logger.info(f"--- Checkpoint: {completed}/{total} references fetched ---")

                    # Rate limiting - delay between API calls
                    time.sleep(config.REQUEST_DELAY)

                except Exception as e:
                    logger.error(f"[{completed}/{total}] ✗ Exception fetching refs for {arxiv_id}: {e}")
                    papers_without_refs.append(arxiv_id)

        stats.end_phase()
        stats.save_checkpoint()

        logger.info(f"\n{'='*80}")
        logger.info(f"Reference extraction completed!")
        logger.info(f"  With references: {len(papers_with_refs)}/{total}")
        logger.info(f"  Without references: {len(papers_without_refs)}/{total}")
        logger.info(f"{'='*80}\n")

        # Mark all successfully processed papers as complete
        for arxiv_id in downloaded_papers:
            if download_results.get(arxiv_id):
                # Mark as complete regardless of reference success
                # some papers legitimately have no references in Semantic Scholar
                # because they are very new or not indexed
                stats.record_paper_success(0)  # Time already recorded during download

        # Update metadata for all papers
        updater = MetadataUpdater(papers_dir=config.PAPERS_DIR)
        updater.process_all_papers(delete_empty=False)

        # Cleanup temp directory
        logger.info("\n--- Cleaning up temporary files ---")
        try:
            if config.TEMP_DIR.exists():
                import shutil
                shutil.rmtree(config.TEMP_DIR)
                config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
                logger.info("Success: Cleaned up temp directory")
        except Exception as e:
            logger.warning(f"Could not clean temp directory: {e}")

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed!")
        logger.info("=" * 80)
        logger.info(f"\nSummary:")
        logger.info(f"  Total papers: {len(papers)}")
        logger.info(f"  Already complete: {len(papers) - len(papers_to_process)}")
        logger.info(f"  Downloaded: {len(downloaded_papers)}")
        logger.info(f"  Failed downloads: {len(failed_downloads)}")
        logger.info(f"  With references: {len(papers_with_refs)}")
        logger.info(f"  Without references: {len(papers_without_refs)}")

    finally:
        stop_monitoring.set()
        monitor_thread.join()
        stats.end()
        stats.save_to_file(config.DATA_DIR / 'statistics_final.json')
        stats.save_checkpoint()
        stats.print_summary()
        logger.info(f"\nStatistics: {config.DATA_DIR / 'statistics_final.json'}")
        logger.info(f"Papers: {config.PAPERS_DIR}")


if __name__ == "__main__":
    main()