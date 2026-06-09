import argparse
import pandas as pd
import tiktoken


DATASET_HEADERS = {
    "miras-sparrow": ["content"],
    "farstail": ["premise", "hypothesis"],
    "farexstance": ["claim", "title"]
}


def compute_token_length(dataset, base_prompt, model_name):
    dataset_path = f"./datasets/{dataset}.csv"
    data = pd.read_csv(dataset_path)
    encoder = tiktoken.encoding_for_model(model_name)
    result = []
    for index, row in data.iterrows():
        if len(DATASET_HEADERS[dataset]) == 1:
            final_prompt = base_prompt + "\n" + row[DATASET_HEADERS[dataset][0]]
        elif len(DATASET_HEADERS[dataset]) == 2:
            final_prompt = base_prompt + "\n" + row[DATASET_HEADERS[dataset][0]] + row[DATASET_HEADERS[dataset][1]]
        else:
            raise ValueError
        token_length = len(encoder.encode(final_prompt))
        result.append(token_length)
    return result


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--dataset", type=str, default="miras-sparrow")
    parser.add_argument("--model_name", type=str, default="gpt-4")
    parser.add_argument("--base_prompt")
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model_name
    base_prompt = args.base_prompt

    result = compute_token_length(dataset, base_prompt, model_name)

    print(f"some of tokens: {sum(result)}")
    print(f"average tokens: {sum(result)/len(result)}")


if __name__ == '__main__':
    main()