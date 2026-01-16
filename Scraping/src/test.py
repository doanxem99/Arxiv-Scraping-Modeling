from lib.statistics_tracker import StatisticsTracker

if __name__ == "__main__":
    stats = StatisticsTracker()
    stats.load_checkpoint()
    stats.print_summary()

