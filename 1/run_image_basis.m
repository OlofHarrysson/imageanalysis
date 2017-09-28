for i = 1:size(stacks{1}, 3)
    [up, error_norm] = image_basis(stacks{1}(:,:,i), bases{1}(:,:,i))
end;