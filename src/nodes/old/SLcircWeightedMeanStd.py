
# SLcircWeightedMeanStd.m
# 
#    date: 131007 last modif 160710
#  author: Steeve Laquitaine
# purpose: calculate circular data statistics (mean, var, std, sem)
# 
#   usage:
# 
#      y are cartesian coordinates
#      data = SLcircWeightedMeanStd([0,1 ; 1,0],[0.5; 0.5])
#  
#      y are polars in deg
#      data = SLcircWeightedMeanStd([0 ; 90],[0.5; 0.5])

function [data,datadegmean,datacoordmean,datadegvar,datadegstd,...
    datadegsem,datadegall,datacoordall] = SLcircWeightedMeanStd(y,p,varargin)

p = SLmakeColumn(p);

# case y is polar (deg) convert to cartesian 
if size(y,2)==1 || size(y,2) > 2
    y = SLmakeColumn(y);
    radius = ones(size(y,2));
    coord = SLpolar2cartesian(y,radius,'polar');
else
    coord = y; # cartesian 
end

data.coord.all  = coord; # cartesian
data.deg.all    = SLgetangle(coord(:,1),coord(:,2));# cartesian to angles (degree)

# calculate mean
data.coord.mean = sum(p(:,ones(1,2)).*data.coord.all); # mean's cartesian & degree
data.deg.mean   = SLgetangle(data.coord.mean(:,1),data.coord.mean(:,2));

# std
data.num        = numel(data.deg.all);
data.deg.allforstd = data.deg.all;
data.deg.meanforstd = repmat(data.deg.mean,data.num,1); # std (degree).

# mean is between 0 and 180.
if data.deg.mean+180 <= 360
    
    # if estimation is>=mean direction+180
    for i = 1 : data.num
        if data.deg.all(i) >= data.deg.mean+180
            data.deg.allforstd(i) = data.deg.all(i) - 360;
        end
    end
    
    # mean is between 180 and 360.
else
    # if the estimated direction sampled is <=the mean direction - 180
    for i = 1 : data.num
        if data.deg.all(i) <= data.deg.mean - 180
            data.deg.meanforstd(i) = data.deg.mean - 360;
        end
    end
end

# var, std and sem
data.deg.var = sum(p.*((data.deg.allforstd - data.deg.meanforstd).^2));
data.deg.std = sqrt(data.deg.var);
data.deg.sem = data.deg.std/sqrt(data.num);

# output variables not within a structure
datadegall = data.deg.all;
datacoordall = data.coord.all;
datadegmean = data.deg.mean;
datacoordmean = data.coord.mean;

# var, std and sem
datadegvar = data.deg.var;
datadegstd = data.deg.std;
datadegsem = data.deg.sem;


# SLpolar2cartesian.m
# 
#    Date: 131007  last modified: 150722
#  Author: Steeve Laquitaine
# Purpose: Convert angle (in deg or rads) to cartesian coordinates (x,y)
# 
# 
# Inputs:
# 
#      theta :  angle
#          r :  unit circle radius
# 
#  Usage:
# 
#       coord = SLpolar2cartesian([1:1:360]',1)
# 
# 
# varargin:
# 
#            'polar'
#            'radian'


function [coord,theta2] = SLpolar2cartesian(theta,r,varargin)

# check args
if nargin <2    
    fprintf('\n (SLpolar2cartesian) Please indicate the radius    \n')
    keyboard    
end

# convert polar to radian
if sum(strcmp(varargin,'polar')) || isempty(varargin)
    
    theta = SLmakeColumn(theta);    # column vector
    theta2.deg = theta;             # angle    
    # theta2.rad = single(theta2.deg)*pi/180; # from degree to radian -->
    # produces negative 0 for 90 deg direction (bad) due to less precision
    theta2.rad = theta2.deg*pi/180; # from degree to radian
    
    # verbose
    if sum(strcmp(varargin,'verbose'))

        fprintf('(SLpolar2cartesian) Assuming input angles are in deg.  \n')
        fprintf('(SLpolar2cartesian) Theta converted to rads then to carts. \n')
    
    end

elseif sum(strcmp(varargin,'radian'))
    
    theta2.deg = SLde2r(theta,0);   # degrees
    theta2.rad = theta;             # radians
    
    # verbose
    if sum(strcmp(varargin,'verbose'))
        fprintf('(SLpolar2cartesian) Theta is in radian and is converted to cartesian.  \n')
    end
end

# Convert radians to cartesian
theta2.rad = SLmakeColumn(theta2.rad);
r = SLmakeColumn(r);
x = r.*cos(theta2.rad);
y = r.*sin(theta2.rad);

# round x and y to 10-4 to prevent weird angle due to imprecision
x = round(x*10^4)/10^4;
y = round(y*10^4)/10^4;
coord = [x y];
