%Build spatial pyramids for the scenes categories dataset
%%Run the file cell wise

image_dir = '../Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign2D';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

%Compute sift descriptors with a grid spacing of 8 and patch size of 16
 %GenerateSiftDescriptors( data.filenames,image_dir,data_dir,params,canSkip,pfig);
%Compute keypoint sift descriptors using vl_feat, and normalize the
%descriptors

%pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT'));
%GenerateKeypointSiftDescriptors( data.filenames,image_dir,data_dir,[],1,pfig);
%close(pfig)


%For me the sift descriptors are already computed, so I'll focus on
%changing the dictionary size and computing the pyramids. The assignment of
%textons to the dictionary items happens while computing histograms, and
%that data is stored in the data directory


%Parameters to skip the building of various stuff, if it already exists in
%the data directory.
%IF you want the stuff recomputed, set the corresponding flag to false
skip_rebuild_dictionaries = 1;
skip_pyramid_rebuild = 1;
skip_rebuild_textons = skip_rebuild_dictionaries; %Change when dictionories are changed
skip_compute_interest_points = 1;


%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
%%
%Compute pyramids and histograms for different dictionary sizes
dictionarySize = [50, 100, 200, 400];
%and pyramid heights
pyramidLevels = [2, 3, 4];

for k = 1:length(dictionarySize)
    pfig = sp_progress_bar(sprintf('Computing With Dic Size %d',dictionarySize(k)));
    params.dictionarySize = dictionarySize(k);
    CalculateDictionary(data.filenames,image_dir,data_dir,'_sift.mat',params, skip_rebuild_dictionaries, pfig);
    BuildHistograms(data.filenames,image_dir,data_dir,'_sift.mat',params, skip_rebuild_textons, pfig);
    for l = 1:length(pyramidLevels)
        fprintf('Dictionary %d Level %d\n', dictionarySize(k), pyramidLevels(l));
        params.pyramidLevels = pyramidLevels(l);
        CompilePyramid(data.filenames,data_dir,sprintf('_texton_ind_%d.mat',params.dictionarySize),params,skip_pyramid_rebuild,pfig);
    end
    close(pfig)
end

%%
%Build Joint Keypoint Neighbourhood Pooling Features for 15 Scenes Dataset

%If changing the parameters for SIFT computation, make sure the parameters
%are reflected here as well
%
params.keypointDetector = 'DoG';
interest_points = ComputeInterestPoints(data.filenames, image_dir, data_dir, params, 100, skip_compute_interest_points);
dictionarySize = [400]; %[50, 100, 200, 400];
sigmaFactors = [3 7 8];%[2 4 6; 2 4 8; 2 4 10; 1 4 6; 1 4 8; 1 4 10; 3 6 8; 3 7 8];
numKeypoints = [30]%[10 20 30];
for i = 1: length(dictionarySize)
    for j = 1: length(numKeypoints)
        for k = 1:size(sigmaFactors,1)
            params.dictionarySize = dictionarySize(i);
            params.sigmaFactors = sigmaFactors(k,:); 
            params.sigmaKeypoints = numKeypoints(j);
%pfig = sp_progress_bar(sprintf('Computing With Dic Size %d',100));
            CompileSigmaSlice( data.filenames,data_dir, sprintf('_texton_ind_%d.mat',params.dictionarySize), interest_points, params, 1);
        end
    end
end


%%
% Build Features for Keypoint Neighbourhood Pooling
%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
% First cluster the keypoint sifts and create textons for those
dictionarySize = [50, 100, 200, 400];
for k = 1:length(dictionarySize)
    pfig = sp_progress_bar(sprintf('Computing With Dic Size %d',dictionarySize(k)));
    params.dictionarySize = dictionarySize(k);
    CalculateDictionary(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',params, 1, pfig);
    BuildHistograms(data.filenames,image_dir,data_dir,'_keypoint_sift.mat',params, 1, pfig);
    close(pfig)
end

%% Now we compile dense descritptors in the neighborhood of keypoints and
%store them in the data directory
params.sigmaFactors = [6 10;                %%%%This is Radius not Dia%%%%
                       6 14;
                       8 16;
                       10 16;
                       10 18
                       ];
%This is used to fetch the scale and locations of SIFT Keypoints
params.keypointFeatureSuffix = '_keypoint_sift.mat';
dictionarySize = [50, 100, 200, 400];
for k = 1:1%length(dictionarySize) %This is really an iteration through dictionary sizes of dense sift. 
                                 %The dictionary size of keypoint sift does
                                 %not come into play here
    params.dictionarySize = dictionarySize(k);
    fprintf('Computing Keypoint Neighbourhood dense pooling for dense dictionary size %d\n', params.dictionarySize);
    ComputeNeighborhoodPoolingHistogram( data.filenames,data_dir, sprintf('_texton_ind_%d.mat',params.dictionarySize), params, 0);
end
%% Next up create dictionaries for these neighbourhood histograms and assign
%labels to them
sigmaFactors = [6 10;
               6 14;
               8 16;
               10 16;
               10 18];
neighborhoodHistogramDictionarySize = [50, 100, 200, 400];
dictionarySize = [20, 50, 80, 100, 120];
params.numTextonImages = 200;
for k = 1: length(neighborhoodHistogramDictionarySize)
    for sf = 1:size(sigmaFactors,1)
        for dic = 1:length(dictionarySize)
            pfig = sp_progress_bar(sprintf('Computing With SF %d_%d HistDic %d Dic Size %d',sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k),dictionarySize(dic)));
            params.dictionarySize = dictionarySize(dic);
            CalculateHistogramDictionary( data.filenames,data_dir, sprintf('_%d_%d_neighborhood_histsize%d.mat', sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k))  , params, 0, pfig );
            AssignHistogramClusters( data.filenames,data_dir, sprintf('_%d_%d_neighborhood_histsize%d.mat', sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k)) , params, 0, pfig );
            close(pfig)
        end
    end
end
    

% Next up, we would construct 2D Histograms 
sigmaFactors = [6 10;
               6 14;
               8 16;
               10 16;
               10 18];
neighborhoodHistogramDictionarySize = [50, 100, 200, 400];
dictionarySize = [20, 50, 80, 100, 120]; %%%This is the size of the histogram of histograms
keypointDictionarySize = [50, 100, 200, 400];
for j = 1: length(keypointDictionarySize)
    for k = 1: length(neighborhoodHistogramDictionarySize)
      for sf = 1:size(sigmaFactors,1)
        for dic = 1:length(dictionarySize)
            pfig = sp_progress_bar(sprintf('Computing With SF %d_%d HistDic %d Dic Size %d',sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k),dictionarySize(dic)));
            CompileNeighborhoodPoolFeature2DHistogram( data.filenames,data_dir, sprintf('_texton_ind_%d_keypoint_sift',keypointDictionarySize(j)), sprintf('_texton_ind_%d_%d_%d_neighborhood_histsize%d', dictionarySize(dic), sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k)), [keypointDictionarySize(j), dictionarySize(dic)], params, 0, pfig );
            close(pfig)
        end
      end
    end
end
%%


