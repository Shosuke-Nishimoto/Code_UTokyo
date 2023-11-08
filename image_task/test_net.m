function accuracy = test_net(net, filepath)
allImages = imageDatastore(filepath,'LabelSource','foldernames');
[~, testStore] = splitEachLabel(allImages, 0.8,'randomize');
predictedLabels = classify(net, testStore);
accuracy = mean(predictedLabels == testStore.Labels);
end