from tvb_hksl_split_parser import tvb_hksl_split_parser

if __name__ == "__main__":
    files = [
        "dataset/tvb-hksl-news/split/dev.csv",
        "dataset/tvb-hksl-news/split/train.csv",
        "dataset/tvb-hksl-news/split/test.csv",
    ]

    for file in files:
        parser = tvb_hksl_split_parser(file)
        tokens = parser.get_train_glosses_tokenized()

        # create a dictionary of all tokens and their frequencies
        token_dict = {}
        for token_list in tokens:
            for token in token_list:
                if token in token_dict:
                    token_dict[token] += 1
                else:
                    token_dict[token] = 1

        # sort the dictionary by frequency
        sorted_dict = dict(sorted(token_dict.items(), key=lambda item: item[1], reverse=False))

        # write the dictionary to a csv file
        with open("split_count_info/" + file.split("/")[-1].split(".")[0] + "_token_count.csv", "w") as f:
            for key in sorted_dict:
                f.write(f"{key},{sorted_dict[key]}\n")

        # write reverse sorted dictionary to a csv file
        count_dict = {}
        for item in sorted_dict.items():
            if item[1] in count_dict:
                count_dict[item[1]].append(item[0])
            else:
                count_dict[item[1]] = [item[0]]

        count_dict = dict(sorted(count_dict.items(), key=lambda item: item[0], reverse=False))

        with open("split_count_info/" + file.split("/")[-1].split(".")[0] + "_count_list.csv", "w") as f:
            for key in count_dict:
                f.write(f"{key},{len(count_dict[key])},{count_dict[key]}\n")
