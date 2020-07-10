% Student Number = 14062340
% Used by scripts GMM_EM.m and GMM_EM_Optimise.m - Step 2

function A = getMatrixA(dim1,dim2,dim3,dim4,bias_order)
% This function will calculate the matrix A for bias field complexity
% orders of 0,1,2,3,4. Each column of A evaluates the smooth polynomial
% basis function at all coordinates of a 3D MRI volume.

% calculate number of polynomials in a 3D basis function 
J = ((bias_order+1)/1)*((bias_order+2)/2)*((bias_order+3)/3);
% allocate memory for A
A = zeros(dim4,J);

% denote spatial position in X, Y and Z dimensions

% Z is effectively the outer for loop (i.e loop through all Z coords) 
Z = linspace(-1,1,dim3);
Z = repmat(Z,dim1*dim2,1);
Z = reshape(Z,[],1);

% Y is then the next inner for loop (i.e. loop through all Y coords)
Y = linspace(-1,1,dim2);
Y = repmat(Y,dim1,1);
Y = reshape(Y,[],1);
Y = repmat(Y,dim3,1);

% X is then next inner for loop (i.e. loop through all X coords)
X = linspace(-1,1,dim1);
X = repmat(X',dim2*dim3,1);

if bias_order == 0
    % This is just one column of all ones
    A(:,1) = 1; % 1
elseif bias_order == 1  
    A(:,1) = 1; % 1
    % Calculate columns 2:4 using polynomials ...
    A(:,2) =  X; % X
    A(:,3) =  Y; % Y
    A(:,4) =  Z; % Z
elseif bias_order == 2
    A(:,1) = 1;  % 1
    A(:,2) =  X; % X
    A(:,3) =  Y; % Y
    A(:,4) =  Z; % Z        
    % Calculate columns 5:10 using polynomials ...
    A(:,5)  =  X.*X; % XX
    A(:,6)  =  Y.*Y; % YY
    A(:,7)  =  Z.*Z; % ZZ
    A(:,8)  =  X.*Y; % XY
    A(:,9)  =  Y.*Z; % YZ
    A(:,10) =  X.*Z; % XZ
elseif bias_order == 3
    A(:,1) = 1;  % 1
    A(:,2) =  X; % X
    A(:,3) =  Y; % Y
    A(:,4) =  Z; % Z        
    A(:,5)  =  X.*X; % XX
    A(:,6)  =  Y.*Y; % YY
    A(:,7)  =  Z.*Z; % ZZ
    A(:,8)  =  X.*Y; % XY
    A(:,9)  =  Y.*Z; % YZ
    A(:,10) =  X.*Z; % XZ        
    % Calculate columns 11:20
    A(:,11) = X.*X.*X; % XXX
    A(:,12) = Y.*Y.*Y; % YYY
    A(:,13) = Z.*Z.*Z; % ZZZ
    A(:,14) = X.*Y.*Z; % XYZ
    A(:,15) = X.*X.*Y; % XXY
    A(:,16) = X.*X.*Z; % XXZ
    A(:,17) = X.*Y.*Y; % XYY
    A(:,18) = Y.*Y.*Z; % YYZ
    A(:,19) = X.*Z.*Z; % XZZ
    A(:,20) = Y.*Z.*Z; % YZZ   
elseif bias_order == 4
    A(:,1) = 1;  % 1
    A(:,2) =  X; % X
    A(:,3) =  Y; % Y
    A(:,4) =  Z; % Z        
    A(:,5)  =  X.*X; % XX
    A(:,6)  =  Y.*Y; % YY
    A(:,7)  =  Z.*Z; % ZZ
    A(:,8)  =  X.*Y; % XY
    A(:,9)  =  Y.*Z; % YZ
    A(:,10) =  X.*Z; % XZ        
    A(:,11) = X.*X.*X; % XXX
    A(:,12) = Y.*Y.*Y; % YYY
    A(:,13) = Z.*Z.*Z; % ZZZ
    A(:,14) = X.*Y.*Z; % XYZ
    A(:,15) = X.*X.*Y; % XXY
    A(:,16) = X.*X.*Z; % XXZ
    A(:,17) = X.*Y.*Y; % XYY
    A(:,18) = Y.*Y.*Z; % YYZ
    A(:,19) = X.*Z.*Z; % XZZ
    A(:,20) = Y.*Z.*Z; % YZZ          
    % Calculate columns 21:35
    A(:,21) = X.*X.*X.*X; % XXXX
    A(:,22) = Y.*Y.*Y.*Y; % YYYY
    A(:,23) = Z.*Z.*Z.*Z; % ZZZZ
    A(:,24) = X.*X.*X.*Y; % XXXY
    A(:,25) = X.*X.*X.*Z; % XXXZ
    A(:,26) = X.*X.*Y.*Z; % XXYZ
    A(:,27) = X.*Y.*Y.*Z; % XYYZ
    A(:,28) = X.*Y.*Z.*Z; % XYZZ
    A(:,29) = X.*Z.*Z.*Z; % XZZZ
    A(:,30) = Y.*Y.*Y.*Z; % YYYZ
    A(:,31) = Y.*Y.*Z.*Z; % YYZZ
    A(:,32) = Y.*Z.*Z.*Z; % YZZZ
    A(:,33) = X.*X.*Z.*Z; % XXZZ
    A(:,34) = X.*X.*Y.*Y; % XXYY
    A(:,35) = X.*Y.*Y.*Y; % XYYY
end

end

