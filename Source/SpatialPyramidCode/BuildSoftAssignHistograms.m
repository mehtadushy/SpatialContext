function [ H_all ] = BuildSoftAssignHistograms( imageFileList,imageBaseDir, dataBaseDir, featureSuffix, params, canSkip, pfig )
%function [ H_all ] = BuildHistograms( imageFileList, dataBaseDir, featureSuffix, params, canSkip )
%
% Find top 3 texton labels of patches with weights and compute texton histograms of all images
%   
% For each image the set of sift descriptors is loaded and then each
%  descriptor is labeled with its texton label. Then the global histogram
%  is calculated for the image. If you wish to just use the Bag of Features
%  image descriptor you can stop at this step, H_all is the histogram or
%  Bag of Features descriptor for all input images.
%
% imageFileList: cell of file paths
% imageBaseDir: the base directory for the image files
% dataBaseDir: the base directory for the data files that are generated
%  by the algorithm. If this dir is the same as imageBaseDir the files
%  will be generated in the same location as the image file
% featureSuffix: this is the suffix appended to the image file name to
%  denote the data file that contains the feature textons and coordinates. 
%  Its default value is '_sift.mat'.
% dictionarySize: size of descriptor dictionary (200 has been found to be
%  a good size)
% canSkip: if true the calculation will be skipped if the appropriate data 
%  file is found in dataBaseDir. This is very useful if you just want to
%  update some of the data or if you've added new images.

fprintf('Building Histograms\n\n');

%% parameters

if(~exist('params','var'))
    params.maxImageSize = 1000;
    params.gridSpacing = 8;
    params.patchSize = 16;
    params.dictionarySize = 200;
    params.numTextonImages = 50;
    params.pyramidLevels = 3;
end
if(~isfield(params,'maxImageSize'))
    params.maxImageSize = 1000;
end
if(~isfield(params,'gridSpacing'))
    params.gridSpacing = 8;
end
if(~isfield(params,'patchSize'))
    params.patchSize = 16;
end
if(~isfield(params,'dictionarySize'))
    params.dictionarySize = 200;
end
if(~isfield(params,'numTextonImages'))
    params.numTextonImages = 50;
end
if(~isfield(params,'pyramidLevels'))
    params.pyramidLevels = 3;
end
if(~exist('canSkip','var'))
    canSkip = 1;
end
%% load texton dictionary (all texton centers)

inFName = fullfile(dataBaseDir, sprintf('dictionary_%d%s', params.dictionarySize, featureSuffix));
load(inFName,'dictionary');
fprintf('Loaded texton dictionary: %d textons\n', params.dictionarySize);

%% compute texton labels of patches and whole-image histograms
H_all = [];
if(exist('pfig','var'))
    tic;
end
for f = 1:length(imageFileList)

    imageFName = imageFileList{f};
    [dirN base] = fileparts(imageFName);
    baseFName = fullfile(dirN, base);
    inFName = fullfile(dataBaseDir, sprintf('%s%s', baseFName, featureSuffix));
    
    if(mod(f,100)==0 && exist('pfig','var'))
        %sp_progress_bar(pfig,3,4,f,length(imageFileList),'Building Soft Assign Histograms:');
    end
    outFName = fullfile(dataBaseDir, sprintf('%s_soft_texton_ind_%d%s', baseFName, params.dictionarySize, featureSuffix));
    outFName2 = fullfile(dataBaseDir, sprintf('%s_soft_hist_%d%s', baseFName, params.dictionarySize, featureSuffix));
    if(exist(outFName,'file')~=0 && exist(outFName2,'file')~=0 && canSkip)
        %fprintf('Skipping %s\n', imageFName);
        if(nargout>1)
            load(outFName2, 'H');
            H_all = [H_all; H];
        end
        continue;
    end
    

    load(inFName, 'features');
    
    ndata = size(features.data,1);
    if(exist('pfig', 'var'))
        %sp_progress_bar(pfig,3,4,f,length(imageFileList),'Building Soft Assign Histograms:');
    end
    %fprintf('Loaded %s, %d descriptors\n', inFName, ndata);

    %% find texton indices and compute histogram 
    texton_ind.data = zeros(ndata,3);
    texton_ind.weights = zeros(ndata,3);
    texton_ind.x = features.x;
    texton_ind.y = features.y;
    texton_ind.wid = features.wid;
    texton_ind.hgt = features.hgt;
    %run in batches to keep the memory foot print small
    batchSize = 100000;
    if ndata <= batchSize
        dist_mat = sp_dist2(features.data, dictionary);
        [min_dist, min_ind] = sort(dist_mat, 2, 'ascend');
        texton_ind.data = min_ind(:,1:3);
        texton_ind.weights = exp(-min_dist(:,1:3)/0.1);
    else
        for j = 1:batchSize:ndata
            lo = j;
            hi = min(j+batchSize-1,ndata);
            dist_mat = sp_dist2(features.data(lo:hi,:), dictionary);
            [min_dist, min_ind] = sort(dist_mat, 2, 'ascend');
            texton_ind.data(lo:hi,:) = min_ind(:,1:3);
            texton_ind.weights(lo:hi,:) = exp(-min_dist(:,1:3)/0.1);

        end
    end
    texton_ind.weights = texton_ind.weights ./ repmat(sum(texton_ind.weights,2), 1, 3);
    H = zeros(1, params.dictionarySize);
    for j = 1:ndata
        H(texton_ind.data(j,1)) = H(texton_ind.data(j,1)) + texton_ind.weights(j,1);
        H(texton_ind.data(j,2)) = H(texton_ind.data(j,2)) + texton_ind.weights(j,2);
        H(texton_ind.data(j,3)) = H(texton_ind.data(j,3)) + texton_ind.weights(j,3);
    end
    H = H/ndata;
    H_all = [H_all; H];
    
    if(features.x ~= texton_ind.x)
        fprintf('Claxon!!!! Woanh woanh woanh!\n'); % Any fans of QI ?
    end

    %% save texton indices and histograms
    sp_make_dir(outFName);
    save(outFName, 'texton_ind');
    save(outFName2, 'H');
end

%% save histograms of all images in this directory in a single file
outFName = fullfile(dataBaseDir, sprintf('soft_histograms_%d%s', params.dictionarySize, featureSuffix));
save(outFName, 'H_all', '-ascii');


end
