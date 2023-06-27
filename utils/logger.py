import logging


class Logger(object):
    def __init__(self, path="log.txt", rank=-1):
        self.logger = logging.getLogger("Logger_" + str(rank))
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        self.file_handler = logging.FileHandler(path, "w")
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(logging.Formatter())
        self.logger.addHandler(self.file_handler)

        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setLevel(logging.INFO if rank in [-1, 0] else logging.WARNING)
        self.stdout_handler.setFormatter(logging.Formatter())
        self.logger.addHandler(self.stdout_handler)

        # Set format (optional)
        # self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        # self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    
    def info(self, text):
        self.logger.info(text)

    def warning(self, text):
        self.logger.warning(text)

    def error(self, text):
        self.logger.error(text)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


if __name__ == '__main__':
    logger_0 = Logger("log_test_0.txt", 0)
    logger_0.info('logger_0 info')
    logger_0.warning('logger_0 warning')

    logger_1 = Logger("log_test_1.txt", 1)
    logger_1.info('logger_1 info')
    logger_1.warning('logger_1 warning')
