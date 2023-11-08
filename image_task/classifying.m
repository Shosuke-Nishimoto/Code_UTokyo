function label = classify(net, filepath)
data = imageDatastore(filepath);
prediction = classify(net,data);
label = char(prediction);
end