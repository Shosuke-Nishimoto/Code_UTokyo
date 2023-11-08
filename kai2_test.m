%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = kai2_test(sample1,sample2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%sample: [data1, data2]

sum1 = sum(sample1);
sum2 = sum(sample2);

sum_data1 = sample1(1) + sample2(1);
sum_data2 = sample1(2) + sample2(2);

sum_all = sum1 + sum2;

expect1 = sum1 ./ sum_all;
expect2 = sum2 ./ sum_all;

expect11 = expect1 * sum_data1;
expect12 = expect1 * sum_data2;
expect21 = expect2 * sum_data1;
expect22 = expect2 * sum_data2;

dif11 = ((sample1(1) - expect11)^2) / expect11;
dif12 = ((sample1(2) - expect12)^2) / expect12;
dif21 = ((sample2(1) - expect21)^2) / expect21;
dif22 = ((sample2(2) - expect22)^2) / expect22;

dif_all = dif11 + dif12 + dif21 + dif22;

p = chi2cdf(dif_all,1,'upper')

