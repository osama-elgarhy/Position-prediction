% Osama Elgarhy (osama.elgarhy@taltech.ee)
% November 2022 - Tallinn, Estonia
% Two Erlang-B calculation functions. To calculate the Blocking probability
% tot_res: The total number of available resources
% Tr: Traffic. Example: Tr=ER*Ct (Entering rate X average crossing time or service time)

%%%%%%%%%%%%%%%%%%
% Example values:
Ct = 30;
ER = 3;
Tr=ER*Ct;
tot_res= 100;
%%%%%%%%%%%%%%%%%%%


bp_sum=0; 
for k=0:tot_res
    bp_sum=bp_sum+((Tr^k)/(factorial(k)));
end
numer=(Tr^tot_res)/(factorial(tot_res));
block_prob_er=numer/bp_sum;
block_prob_1=100*block_prob_er; % Resulting Blocking probability - function 1



block_prob_er_rec=1;
for eri=1:tot_res
    block_prob_er_rec=1+ (block_prob_er_rec)*(eri/Tr);
end
block_prob_2=100*(1/block_prob_er_rec); % Resulting Blocking probability - function 2
