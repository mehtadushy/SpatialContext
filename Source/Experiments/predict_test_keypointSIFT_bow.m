%Using LibSVM Train for Keypoint SIFT BoW
image_dir = '../Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir, 'scenes_names.mat'));
load(fullfile(data_dir, 'category_names.mat'));
%Do a test-train split (data is already randomized
train_percent = 90;
train.index = (1 : floor(length(data.labels)*train_percent/100));
train.labels = data.labels(train.index);
test.index = (length(train.index)+1 : length(data.labels));
test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');

dictionary_size = [50, 100, 200, 400];

%%
% Build Features for Keypoint Neighbourhood Pooling
%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
% First cluster the keypoint sifts and create textons for those
dictionarySize = [50, 100, 200, 400];
for k = 1:length(dictionarySize)
    pfig = sp_progress_bar(sprintf('Computing With Dic Size %d',dictionarySize(k)));
    params.dictionarySize = dictionarySize(k);
    %CalculateDictionary(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',params, 1, pfig);
    BuildHistograms(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',params, 0, pfig);
    close(pfig)
end


%%

%Train SVM with intersection kernel for Keypoint BoW, and predict on the test set
%It is multi-class classification, done one vs one, and same parameters
% used for all discriminants


bow_accuracy = zeros(size(dictionary_size));
parfor i = 1:length(dictionary_size)
    histogram = load(fullfile(data_dir,sprintf('histograms_%d_keypoint_sift.mat',dictionary_size(i))),'-ascii');
    dat = load(train_test_datafilename);
    train = dat.train;
    test = dat.test;
    train.features = histogram(train.index,:);
    test.features = histogram(test.index,:);

    %We'll do multiclass classification
    %Compute kernel for training data
    train.K = [(1:size(train.features,1))' , intersectionKernel(train.features, train.features)];
    test.K = [(1:size(test.features,1))' , intersectionKernel(test.features, train.features)];
    
    bow_model = svmtrain(train.labels',train.K, '-t 4' );
    [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
    bow_accuracy(i) = accuracy_P(1);
end

outfname = fullfile( data_dir, 'results', 'keypointSift_bow_accuracy.mat');
save(outfname, 'bow_accuracy', 'dictionary_size');