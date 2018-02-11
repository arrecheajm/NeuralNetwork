class Logger(object):

    def __init__(self, file_name, append):

        if file_name is not None:
            self.fileName = file_name
        if append:
            self.file = open(file_name, 'a')
        else:
            self.file = open(file_name, 'w')

    def log_line(self, text):
        self.file.write(text)

    def close(self):
        self.file.close()

