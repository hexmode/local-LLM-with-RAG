from models import check_if_model_is_available
from document_loader import load_documents_into_database
import argparse
import sys

from llm_graph import getChatGraph


def main() -> None:
    args = parse_arguments()
    llm_model_name = args.model
    embedding_model_name = args.embedding_model
    documents_path = args.path

    # Check to see if the models available, if not attempt to pull them
    try:
        check_if_model_is_available(llm_model_name)
        check_if_model_is_available(embedding_model_name)
    except Exception as e:
        print(e)
        sys.exit()

    # Creating database form documents
    try:
        db = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    chat = getChatGraph(llm_model_name, db)

    while True:
        try:
            user_input = input(
                "\n\nPlease enter your question (or type 'exit' to end): "
            ).strip()
            if user_input.lower() == "exit":
                break
            for event in chat.stream({"messages": [{"role": "user", "content": user_input}]}):
                for value in event.values():
                    print("Assistant:", value["messages"][-1].content)

        except KeyboardInterrupt:
            break


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LLM with RAG with Ollama.")
    parser.add_argument(
        "-m",
        "--model",
        default="mistral",
        help="The name of the LLM model to use.",
    )
    parser.add_argument(
        "-e",
        "--embedding_model",
        default="nomic-embed-text",
        help="The name of the embedding model to use.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default="Research",
        help="The path to the directory containing documents to load.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
