%% Keypoint SIFT BoW With Soft Assign Histogram
clear
%image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_RotImages3';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign2D';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'rot3_scenes_names.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Flags
DoG_keypoints = false; % uses DoG keypoints for SIFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params.SIFTMagnif = 3;
params.axisAlignedSIFT = false; 
aas = false;   %<--  ugly hack
pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT'));

if(DoG_keypoints==true)
    GenerateDoGKeypointSiftDescriptors( data.filenames,image_dir,data_dir,params,0,pfig);
else
    GenerateKeypointSiftDescriptors( data.filenames,image_dir,data_dir,params,0,pfig);
end
close(pfig)


%Do a test-train split (data is already randomized
% train_percent = 90;
% train.index = (1 : floor(length(data.labels)*train_percent/100));
% train.labels = data.labels(train.index);
% test.index = (length(train.index)+1 : length(data.labels));
% test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'rot3_train_test_data.mat');
save(train_test_datafilename, 'train', 'test');

dictionary_size = [50, 100, 200, 400, 600, 800, 1000, 1500];

%%
% Build Features for Keypoint Neighbourhood Pooling
%Pick 1000 images to do k_means on
%params.numTextonImages = 1000;
clear params;
% First cluster the keypoint sifts and create textons for those
dictionarySize = [50, 100, 200, 400, 600, 800, 1000, 1500];
parfor k = 1:length(dictionarySize)
    params = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
    %pfig = sp_progress_bar(sprintf('Computing With Dic Size %d',dictionarySize(k)));
    %params.dictionarySize = dictionarySize(k);
    CalculateDictionary(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',params, 0);
    BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',params, 0);
    %close(pfig)
end


%%

%Train SVM with intersection kernel for Keypoint BoW, and predict on the test set
%It is multi-class classification, done one vs one, and same parameters
% used for all discriminants


bow_accuracy = zeros(size(dictionary_size));
parfor i = 1:length(dictionary_size)
    histogram = load(fullfile(data_dir,sprintf('soft_histograms_%d_keypoint_sift.mat',dictionary_size(i))),'-ascii');
    dat = load(train_test_datafilename);
    train = dat.train;
    test = dat.test;
    train.features = histogram(train.index,:);
    test.features = histogram(test.index,:);

    %We'll do multiclass classification
    %Compute kernel for training data
    train.K = [(1:size(train.features,1))' , intersectionKernel(train.features, train.features)];
    test.K = [(1:size(test.features,1))' , intersectionKernel(test.features, train.features)];
    
            bestcv = 0;
            for log2c = -2:2:4
                params = sprintf('-q -c %d -v 2 -t 4', 2^log2c);
                cv = svmtrain(train.labels',train.K, params );
                if (cv >= bestcv),
                  bestcv = cv; bestc = 2^log2c; bestlog2c = log2c;
                end
            end
            fprintf('Best C = %d\n',bestc);
            bestcv = 0;
            for log2c = (bestlog2c-0.3):0.1:(bestlog2c+0.3)
                params = sprintf('-q -c %d -v 2 -t 4', 2^log2c);
                cv = svmtrain(train.labels',train.K, params );
                if (cv >= bestcv),
                  bestcv = cv; C = 2^log2c;
                end
            end
            fprintf('C = %d\n',bestc);
            params = sprintf('-q -c %d -t 4', C);
            bow_model = svmtrain(train.labels',train.K, params );
    [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
    bow_accuracy(i) = accuracy_P(1);
end
if(DoG_keypoints==true)
 keypt_type = 'DoG_';
else
  keypt_type = '';
end
if(aas==true)
 a_a = 'AxisAligned_';
else
  a_a = '';
end
p = true; 
outfname = fullfile( data_dir, 'results', sprintf('%s%skeypointSift_mag%d_bow_accuracy.mat', keypt_type, a_a, 3));
save(outfname, 'bow_accuracy', 'dictionary_size');
