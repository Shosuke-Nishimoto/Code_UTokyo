function trainStoreAug = makeData(filepath)
% データをimageDatastoreとして持つ
allImages = imageDatastore(filepath,'LabelSource','foldernames');

% データを分割
%[trainStore, testStore] = splitEachLabel(allImages, 0.8,'randomize');

% データの水増し
allImages = shuffle(allImages);
bootstrap_factor = 2;
alphabetical_labels = {'class1','class2','class3','class4','class5'};
labels = allImages.Labels;
labelCounts = countEachLabel(allImages);
labelCounts = labelCounts.Count;
weights = labelCounts/sum(labelCounts);
weights = weights.^(-1);
weightVec = [];
for lab = 1:length(labels)
  for labidx = 1:length(alphabetical_labels)
    if labels(lab) == alphabetical_labels(labidx)
      weightVec(lab) = weights(labidx);
    end
  end
end
trainFiles = allImages.Files;
bootstrapSize = round(length(trainFiles) * bootstrap_factor);
Bootstrap = datasample(trainFiles, bootstrapSize, 'Weights', weightVec);
bootStrapTrainStore = imageDatastore(Bootstrap, 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

% 訓練データの整形
augmentedResolution = [40 40];
augmenter = imageDataAugmenter('RandRotation', [-10 10]);
trainStoreAug = augmentedImageDatastore(augmentedResolution, bootStrapTrainStore, 'DataAugmentation', augmenter);

end