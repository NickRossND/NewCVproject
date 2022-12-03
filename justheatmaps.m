clear all;
close all;
%% load files

files = dir("DK*_*");
for aa=1:12
    tic
    patient = convertCharsToStrings(files(aa).name);
    datapath = "G:\My Drive\Notre Dame\Classes\ComputerVision\CVproject-BreastCancer\SandhyaData\DOSI_data\PROCESSED\"+patient+"\";
    savepath = "G:\My Drive\Notre Dame\Classes\ComputerVision\CVproject-BreastCancer\SandhyaData\CVProject\PatientPlots\"+patient+"\";
    
    checkLeft = (datapath + 'pt'+patient+'_sv_L_v3__CHROM_SS.asc');
    
    if exist(checkLeft) == 2
        Left = importdata(datapath + 'pt'+patient+'_sv_L_v3__CHROM_SS.asc');
        Right = importdata(datapath + 'pt'+patient+'_sv_R_v3__CHROM_SS.asc');
        v3list(aa,1) = aa;
    else
        checkLeft = (datapath + 'pt'+patient+'_sv_L_v3__CHROM_SS.asc');
        if exist(checkLeft) == 2
            Left = importdata(datapath + 'pt'+patient+'_sv_L_v2__CHROM_SS.asc');
            Right = importdata(datapath + 'pt'+patient+'_sv_R_v2__CHROM_SS.asc');
            v3list(aa,2) = aa;
        else
            continue;
        end
    end
    disp(aa);
    

%HbO2
%Hb
%H2O
%MetHb
%fat

%% Set file names

    names = ["Right-HbO2", "Right-Hb", "Right-H2O", "Right-Fat", "Right-THC","Right-O2sat", "Right-TOI"; "Left-HbO2", "Left-Hb", "Left-H2O", "Left-Fat","Left-THC","Left-O2sat","Left-TOI"];
    
    % 
    %% 
    % Left
    k=5; %column position where chromphores start
    a=1;
    for i=2:size(Left.textdata,1) %% looping through all grid points
        str=Left.textdata(i);
        %%converting cell to string
        B=str{1};
        if B(2)=='N' 
            Color(a,1)=-str2num(B(3:5));
        else
            Color(a,1)=str2num(B(3:5));
        end
        if B(6)=='N'
            Color(a,2)=-str2num(B(7:9));
        else
            Color(a,2)=str2num(B(7:9));
        end
        Color(a,3)=Left.data(i-1,k);
        Color(a,4)=Left.data(i-1,k+2);
        Color(a,5)=Left.data(i-1,k+4);
        Color(a,6)=Left.data(i-1,k+6);
        Color(a,7)=Left.data(i-1,k+8);
        Color(a,8)=Left.data(i-1,k+10);
        Color(a,9)=Left.data(i-1,k+12);
        a=a+1;
    end
    %%
    %sort whole matrix by x and y coord. so that the information can be plotted & saved properly
    Color = sortrows(Color);
    X = Color(:,1);
    Y = Color(:,2);
    x = [min(X),max(X)];
    y = [min(Y),max(Y)];
    
    %remove the X and Y coordinates from the matrix
    Color(:,1:2)=[];
    
    
    %%
    % Right 
    
    k=5; %column position where chromphores start
    a=1;
    for i=2:size(Right.textdata,1) %% looping through all grid points
        str=Right.textdata(i);
        %%converting cell to string
        C=str{1};
        if C(2)=='N'
            Color2(a,1)=-str2num(C(3:5));
        else
            Color2(a,1)=str2num(C(3:5));
        end
        if C(6)=='N'
            Color2(a,2)=-str2num(C(7:9));
        else
            Color2(a,2)=str2num(C(7:9));
        end
        Color2(a,3)=Right.data(i-1,k);
        Color2(a,4)=Right.data(i-1,k+2);
        Color2(a,5)=Right.data(i-1,k+4);
        Color2(a,6)=Right.data(i-1,k+6);
        Color2(a,7)=Right.data(i-1,k+8);
        Color2(a,8)=Right.data(i-1,k+10);
        Color2(a,9)=Right.data(i-1,k+12);
        a=a+1;
    end
    Color2 = sortrows(Color2);
    X2 = Color2(:,1);
    Y2 = Color2(:,2);
    xx2 = [min(X2),max(X2)];
    yy2 = [min(Y2),max(Y2)];
    
    Color2(:,1:2)=[];
    
    
    %% plot chomophores
    x_len = length(unique(X));
    y_len = length(unique(Y));
    
    x2_len = length(unique(X2));
    y2_len = length(unique(Y2));
    
    
    for j=1:7
        minimum_left = min(Color(:,j), [], 'all');
        minimum_right = min(Color2(:,j), [], 'all');
        maximum_left = max(Color(:,j), [], 'all');
        maximum_right = max(Color2(:,j), [], 'all');
        
        if minimum_left < minimum_right
            abs_min = minimum_left;
        else
            abs_min = minimum_right;
        end
        
        if maximum_left > maximum_right
            abs_max = maximum_left;
        else
            abs_max = maximum_right;
        end
    
        figure(j);
        Cnew = reshape(Color(:,j),y_len,x_len);
        left = imagesc(x,y,Cnew);
        axis xy;
        colormap jet;
        colorbar;
        caxis([abs_min, abs_max]);
        
        figure(j+7);
        Cnew2 = reshape(Color2(:,j),y2_len,x2_len);
        right = imagesc(-1*xx2,yy2,Cnew2);
        axis xy;
        colormap jet;
        colorbar;
        caxis([abs_min, abs_max]);
        
        Cnew = flip(Cnew,1);
        Cnew2 = flip(Cnew2,1);
        saveas(left,savepath+names(2,j)+".png")
        saveas(right,savepath+names(1,j)+".png")
        save(savepath+names(2,j)+'.txt','Cnew','-ascii')
        save(savepath+names(1,j)+'.txt','Cnew2','-ascii')
    end
    clearvars -except files;
    close all;
    toc
end



