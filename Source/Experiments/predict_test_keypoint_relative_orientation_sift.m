%Using LibSVM Train
%This file too is to be run cell wise
clear;
image_dir = '../Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign_2D';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));
load(fullfile(data_dir2, 'category_names.mat'));
%Do a test-train split (data is already randomized
train_percent = 90;
train.index = (1 : floor(length(data.labels)*train_percent/100));
train.labels = data.labels(train.index);
test.index = (length(train.index)+1 : length(data.labels));
test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');
%%

%Pick 1000 images to do k_means on
params.numTextonImages = 1000;

neighbourhoodSize =   [
                       40;
                       80;
                       ];
dictionarySize = [50, 100, 200, 400, 600, 800, 1000, 1500];
              %1500, 2000, 2500];

accuracy_results = cell( length(dictionarySize)+1, length(neighbourhoodSize)+1);


for k = 1:length(dictionarySize)
   params.dictionarySize = dictionarySize(k);
   accuracy_results{ k+1, 1} = sprintf('%d Dic',dictionarySize(k));
   for i = 1:length(neighbourhoodSize)
        accuracy_results{1, i+1} = sprintf('%dN',neighbourhoodSize(i));
   end
end




%%
for i = 1:length(neighbourhoodSize)
   acc_log = cell(length(dictionarySize),1);
     parfor k = 1:length(dictionarySize)

       fprintf('Predicting With %d Dictionary %d Neigh Size', dictionarySize(k), neighbourhoodSize(i));

            inFName = fullfile(data_dir, sprintf('soft_histograms_%d_%d_neigh_ROcontext_keypoint_sift.mat', dictionarySize(k),neighbourhoodSize(i) ));
            H_all = load(inFName,'-ascii');
            dat = load(train_test_datafilename);
            train = dat.train;
            test = dat.test;
            train_features = H_all(train.index,:);
            test_features = H_all(test.index,:);

            %We'll do multiclass classification
            %Compute kernel for training data
            train.K = [(1:size(train_features,1))' , intersectionKernel(train_features, train_features)];
            test.K = [(1:size(test_features,1))' , intersectionKernel(test_features, train_features)];
    
            bow_model = svmtrain(train.labels',train.K, '-t 4' );
            [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
            acc_log{k} = accuracy_P(1);

     end
            accuracy_results(2:end ,i+1) = acc_log;
end


outfname = fullfile( data_dir, 'results', 'ROcontext_sift_accuracy.mat');
save(outfname, 'accuracy_results');







