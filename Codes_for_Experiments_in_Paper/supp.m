function supp(w,N)

    supp_tmp=zeros(1,N);
    supp_tmp(find(w~=0))=1;
    supp_total=[supp_total;supp_tmp];
    sum_supp_total=sum(supp_total,2);
        
end