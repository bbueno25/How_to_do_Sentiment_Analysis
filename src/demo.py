import tflearn

if __name__ == "__main__":
    train, test, _ = tflearn.datasets.imdb.load_data(
        path='data/imdb.pkl', n_words=10000, valid_portion=0.1
        )

    trainX, trainY = train
    testX, testY = test

    # data preprocessing/sequence padding
    trainX = tflearn.data_utils.pad_sequences(trainX, maxlen=100, value=0.0)
    testX = tflearn.data_utils.pad_sequences(testX, maxlen=100, value=0.0)
    
    # converting labels to binary vectors
    trainY = tflearn.data_utils.to_categorical(trainY, nb_classes=2)
    testY = tflearn.data_utils.to_categorical(testY, nb_classes=2)
    
    # network consruction
    net = tflearn.input_data([None, 100])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(
        net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy'
        )
    
    # training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32)
