from collections import defaultdict


def evaluate_tracie_style():
    glines = [x.strip() for x in open("../../../data/iid/tracie_test.txt").readlines()]
    plines = [x.strip() for x in open("experiment_result_iid/eval_results_lm.txt").readlines()]
    assert len(glines) == len(plines)
    total = 0
    correct = 0
    total_start = 0
    correct_start = 0
    total_end = 0
    correct_end = 0
    story_prediction_map = {}
    pred_map = defaultdict(int)
    with open("error.txt", "w") as f:
        for i, l in enumerate(glines):
            flag = False
            if "story:" in l.split("\t")[0]:
                story = l.split("\t")[0].split("story:")[1]
            else:
                story = "no story"
            if story not in story_prediction_map:
                story_prediction_map[story] = []
            label = l.split("\t")[1].split()[1]
            p = plines[i].split()[1][:8]
            total += 1
            if label == p:
                correct += 1
                story_prediction_map[story].append(True)
            else:
                flag = True
                story_prediction_map[story].append(False)
            if flag:
                if "starts after" in l:
                    pred_map["starts_after_"+label] += 1
                elif "starts before" in l:
                    pred_map["starts_before_" + label] += 1
                elif "ends after" in l:
                    pred_map["ends_after_" + label] += 1
                else:
                    pred_map["ends_before_" + label] += 1
                f.write("story: {}\t query:{} \tpred: {}\t label: {}\n".format(story, l.split("story")[0], p, label))
    for k, v in pred_map.items():
        print("key:{}, value:{}\n".format(k, v))


evaluate_tracie_style()