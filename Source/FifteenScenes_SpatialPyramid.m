% %Build spatial pyramids for the scenes categories dataset
% %%Run the file cell wise
% 
 clear;
 %image_dir = '../Data/SceneCategoriesDataset_RotatedImages';
 image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_RotImages3';
 data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign2D';
 data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
 load(fullfile(data_dir2, 'rot3_scenes_names.mat'));
%%
%Compute sift descriptors with a grid spacing of 8 and patch size of 16

pfig = sp_progress_bar(sprintf('Computing SIFT'));
GenerateSiftDescriptors( data.filenames,image_dir,data_dir,[],0,pfig);
%Compute keypoint sift descriptors using vl_feat, and normalize the
%descriptors

%pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT'));
%GenerateKeypointSiftDescriptors( data.filenames,image_dir,data_dir,[],1,pfig);
close(pfig)


%For me the sift descriptors are already computed, so I'll focus on
%changing the dictionary size and computing the pyramids. The assignment of
%textons to the dictionary items happens while computing histograms, and
%that data is stored in the data directory


%Parameters to skip the building of various stuff, if it already exists in
%the data directory.
%IF you want the stuff recomputed, set the corresponding flag to false
skip_rebuild_dictionaries = 0;
skip_pyramid_rebuild = 0;
skip_rebuild_textons = skip_rebuild_dictionaries; %Change when dictionories are changed
skip_compute_interest_points = 0;


%Pick 1000 images to do k_means on

%%
%Compute pyramids and histograms for different dictionary sizes
dictionarySize = [50, 100, 200, 400, 800, 1000 ];
%and pyramid heights
pyramidLevels = [2, 3];

parfor k = 1:length(dictionarySize)
    pfig = sp_progress_bar(sprintf('Computing With Dic Size %d',dictionarySize(k)));
    params = struct('dictionarySize',dictionarySize(k), 'numTextonImages', 1000);
    CalculateDictionary(data.filenames,image_dir,data_dir,'_sift.mat',params, skip_rebuild_dictionaries, pfig);
    BuildHistograms(data.filenames,image_dir,data_dir,'_sift.mat',params, skip_rebuild_textons, pfig);
    BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,'_sift.mat',params, skip_rebuild_textons, pfig)
    for l = 1:length(pyramidLevels)
        fprintf('Dictionary %d Level %d\n', dictionarySize(k), pyramidLevels(l));
        params.pyramidLevels = pyramidLevels(l);
        CompilePyramid(data.filenames,data_dir,sprintf('_texton_ind_%d_sift.mat',params.dictionarySize),params,skip_pyramid_rebuild,pfig);
    end
    close(pfig)
end

%%
%Do a test-train split (data is already randomized
% train_percent = 90;
% train.index = (1 : floor(length(data.labels)*train_percent/100));
% train.labels = data.labels(train.index);
% test.index = (length(train.index)+1 : length(data.labels));
% test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'rot3_train_test_data.mat');
save(train_test_datafilename, 'train', 'test');
% %%
 dictionary_size = [50, 100, 200, 400, 800, 1000];
 pyramid_levels = [2, 3];

%%

%Train SVM with intersection kernel for BoW, and predict on the test set
%It is multi-class classification, done one vs one, and same parameters
% used for all discriminants
%Cross validation ??

bow_accuracy = zeros(size(dictionary_size));
parfor i = 1:length(dictionary_size)
    histogram = load(fullfile(data_dir,sprintf('histograms_%d_sift.mat',dictionary_size(i))),'-ascii');
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

outfname = fullfile( data_dir, 'bow_accuracy.mat');
save(outfname, 'bow_accuracy', 'dictionary_size');
nonsoft_bow_acc = bow_accuracy;
%%
bow_accuracy = zeros(size(dictionary_size));
parfor i = 1:length(dictionary_size)
    histogram = load(fullfile(data_dir,sprintf('soft_histograms_%d_sift.mat',dictionary_size(i))),'-ascii');
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

outfname = fullfile( data_dir, 'soft_bow_accuracy.mat');
save(outfname, 'bow_accuracy', 'dictionary_size');
%
% Train SVM with intersection kernel for Pyramid Kernels, and predict on the test set
% It is multi-class classification, done one vs one, and same parameters
% used for all discriminants


  pyramid_accuracy = zeros( length(dictionary_size)+1 , length(pyramid_levels)+1);
pyramid_accuracy(2:end,1) = dictionary_size';
pyramid_accuracy(1, 2:end) = pyramid_levels;
for j = 1:length(pyramid_levels)
   pyramid_accuracy_temp = zeros(length(dictionary_size),1);
   parfor i = 1:length(dictionary_size)
       fprintf('Level %d Dictionary Size %d\n', pyramid_levels(j), dictionary_size(i));
       feat = load(fullfile(data_dir,sprintf('pyramids_all_%d_%d.mat',dictionary_size(i), pyramid_levels(j))));
       dat = load(train_test_datafilename);
           train = dat.train;
           test = dat.test;
       train.features = feat.pyramid_all(train.index,:);
       test.features = feat.pyramid_all(test.index,:);

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
       pyramid_accuracy_temp(i) = accuracy_P(1);
   end
   pyramid_accuracy(2:end, j+1) = pyramid_accuracy_temp;
   accuracy = pyramid_accuracy(2:end, j+1);
   %outfname = fullfile( data_dir, sprintf('pyramid_%d_accuracy.mat',pyramid_levels(j)) );
   %save(outfname, 'accuracy');
end

outfname = fullfile( data_dir, 'pyramid_accuracy.mat');
save(outfname, 'pyramid_accuracy');


