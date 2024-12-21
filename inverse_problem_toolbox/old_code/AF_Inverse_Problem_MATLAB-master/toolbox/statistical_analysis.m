function p_value_metric = statistical_analysis (reference_metric,comparison_metric,selected_nodes)
if isempty(selected_nodes)
    if(length(reference_metric)>25 &&...
            length(comparison_metric)>25 &&...
            kstest(reference_metric)&&...
            kstest(comparison_metric))
        [~,p_value_metric]=ttest(reference_metric,comparison_metric);
    else
        p_value_metric=signrank(reference_metric,comparison_metric);
    end
else
    if(length(reference_metric(selected_nodes))>25 &&...
            length(comparison_metric(selected_nodes))>25 &&...
            kstest(reference_metric(selected_nodes))&&...
            kstest(comparison_metric(selected_nodes)))
        [~,p_value_metric]=ttest(reference_metric(selected_nodes),comparison_metric(selected_nodes));
    else
        p_value_metric=signrank(reference_metric(selected_nodes),comparison_metric(selected_nodes));
    end
end
end
