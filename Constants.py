import socket

hostname = socket.gethostname()

TRAIN_BATCH = 500
TEST_BATCH = 100

if hostname.lower() == 'main-pc':
    TRAIN_BATCH = 1500
    TEST_BATCH = 1000

EPOCHS = 90