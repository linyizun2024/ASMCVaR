function [top_m_values, top_m_indices] = maxk(vector, m)

    if m > length(vector)
        error('m exceeds the length of vector');
    end


    [~, sorted_indices] = sort(vector, 'descend');

    top_m_values = vector(sorted_indices(1:m));
    top_m_indices = sorted_indices(1:m);
end

