% Programa para gerar o operador de Koopman usando DMD
% Retirado do livro do J.N. Kutz - Abril/2022 - UFOP

% Dados iniciais
dt = 1;  % Passo temporal
r = 3;   % Rank do operador de Koopman
X1 = [1,2,3,4,5;2,3,4,5,6;3,5,7,9,11]     % Dados prévios
X2 = [2,3,4,5,6;3,4,5,6,7;5,7,9,11,13];   % Dados posteriores

% Decomposição SVD do sistema
[U, S, V] = svd(X1, 'econ');

% Truncamento do sistema de acordo com o rank
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);

% Construção da matriz A_til
Atil = Ur'*X2*Vr/Sr;
% Decomposição da matriz A_til
[Wr, D] = eig(Atil);

% Construção dos autovetores de Koopman
Phi = X2*Vr/Sr*Wr;
Phinv = pinv(Phi);        % Pseudoinverso

% Operadores dos modos dinâmicos
lambda = diag(D);         % Autovalores de A_til
omega = log(lambda)/dt;   % Argumento da solução da EDO associada

% Vetores importantes
x1 = X1(:, 1);  % Estado inicial do sistema
b = Phi\x1;     % Amplitude inicial dos modos dinâmicos

% Quatidade de estados, vetor solução temporal e vetor temporal
m1 = size(X1, 2);
timed = zeros(r, m1);
t = (0:m1-1)*dt;

% Processo de integração - Transformada de Fourier
for it = 1:m1,
  timed(:,it) = (b.*exp(omega*t(it)));
end;

% Reconstrução do espaço de estados
X1d = Phi*timed;
X1_dmd = int8(real(X1d))

% Selecionando o último estado
ue = size(X1,2);
x1f = X1(:,ue);

% Calculando a matriz A referente ao operador K
A = real(Phi*D*Phinv);

% Calculando o próximo estado, que coincide com
% o estado final de X2
x2f = A*x1f

% FIM
