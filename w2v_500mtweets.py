{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9ce69537",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2024-07-18T20:17:21.695000Z",
     "iopub.status.busy": "2024-07-18T20:17:21.694572Z",
     "iopub.status.idle": "2024-07-18T20:17:36.930087Z",
     "shell.execute_reply": "2024-07-18T20:17:36.928596Z"
    },
    "papermill": {
     "duration": 15.243936,
     "end_time": "2024-07-18T20:17:36.932828",
     "exception": false,
     "start_time": "2024-07-18T20:17:21.688892",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to /usr/share/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to /usr/share/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Import necessary libraries\n",
    "from gensim.models import KeyedVectors\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.corpus import stopwords\n",
    "import nltk\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os \n",
    "\n",
    "# Download necessary NLTK data\n",
    "nltk.download('punkt')\n",
    "nltk.download('stopwords')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "66d92e7a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-07-18T20:17:36.942675Z",
     "iopub.status.busy": "2024-07-18T20:17:36.941575Z",
     "iopub.status.idle": "2024-07-18T20:22:11.023110Z",
     "shell.execute_reply": "2024-07-18T20:22:11.021669Z"
    },
    "papermill": {
     "duration": 274.091667,
     "end_time": "2024-07-18T20:22:11.028376",
     "exception": false,
     "start_time": "2024-07-18T20:17:36.936709",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model loaded successfully.\n"
     ]
    }
   ],
   "source": [
    "from gensim.models import KeyedVectors\n",
    "\n",
    "# Update the model path to point to the correct file\n",
    "model_path = '/kaggle/input/word2vec550mtweets-300dim/ntua_twitter_300.txt'\n",
    "\n",
    "# Load the pre-trained Word2Vec model with binary=False\n",
    "model = KeyedVectors.load_word2vec_format(model_path, binary=False)\n",
    "print(\"Model loaded successfully.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "1244d1cd",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-07-18T20:22:11.037786Z",
     "iopub.status.busy": "2024-07-18T20:22:11.037327Z",
     "iopub.status.idle": "2024-07-18T20:22:11.044467Z",
     "shell.execute_reply": "2024-07-18T20:22:11.043271Z"
    },
    "papermill": {
     "duration": 0.014887,
     "end_time": "2024-07-18T20:22:11.047006",
     "exception": false,
     "start_time": "2024-07-18T20:22:11.032119",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#define function to get compound vector if needed \n",
    "def get_compound_vector(word2vec_model, compound_word):\n",
    "        words = compound_word.replace('-',' ').split()\n",
    "        word_vectors = [word2vec_model[word] for word in words if word in word2vec_model]\n",
    "        \n",
    "        if not word_vectors:\n",
    "                return None #None of the words in the compound word are in corpora\n",
    "        compound_vector = np.mean(word_vectors, axis = 0)\n",
    "        return compound_vector "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e8e103a0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-07-18T20:22:11.056602Z",
     "iopub.status.busy": "2024-07-18T20:22:11.056150Z",
     "iopub.status.idle": "2024-07-18T20:22:11.079714Z",
     "shell.execute_reply": "2024-07-18T20:22:11.078477Z"
    },
    "papermill": {
     "duration": 0.031469,
     "end_time": "2024-07-18T20:22:11.082380",
     "exception": false,
     "start_time": "2024-07-18T20:22:11.050911",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Function to compute similarity between two words\n",
    "def compute_similarity(word1, word2):\n",
    "    if word1 in model.key_to_index and word2 in model.key_to_index:\n",
    "        similarity = model.similarity(word1, word2)\n",
    "        return similarity\n",
    "    else:\n",
    "        vector1 = get_compound_vector(model, word1) if word1 not in model.key_to_index else model[word1]\n",
    "        vector2 = get_compound_vector(model, word2) if word2 not in model.key_to_index else model[word2]\n",
    "        if vector1 is not None and vector2 is not None:\n",
    "            similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))\n",
    "            return similarity \n",
    "        else:\n",
    "            missing_words = [word for word in [word1, word2] if word not in model.key_to_index and get_compound_vector(model, word) is None]\n",
    "            return f\"Word(s) not in vocabulary: {', '.join(missing_words)}\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "73baf8f4",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-07-18T20:22:11.092180Z",
     "iopub.status.busy": "2024-07-18T20:22:11.091446Z",
     "iopub.status.idle": "2024-07-18T20:22:11.098371Z",
     "shell.execute_reply": "2024-07-18T20:22:11.096283Z"
    },
    "papermill": {
     "duration": 0.015061,
     "end_time": "2024-07-18T20:22:11.101262",
     "exception": false,
     "start_time": "2024-07-18T20:22:11.086201",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Specify the indiviudal input file paths\n",
    "input_file_paths = [\n",
    "    '/kaggle/input/actions-and-agents-datasets/actions_and_distractors_output.csv',  # Update this path\n",
    "    '/kaggle/input/actions-and-agents-datasets/agents_and_distractors_output.csv'  # Update this path\n",
    "]\n",
    "\n",
    "# Specify the corresponding output file names\n",
    "output_file_names = [\n",
    "    'tweets_output_actions_similarity_scores.csv',\n",
    "    'tweets_output_agents_similairity_scores.csv'\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b7415b6c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-07-18T20:22:11.110799Z",
     "iopub.status.busy": "2024-07-18T20:22:11.110415Z",
     "iopub.status.idle": "2024-07-18T20:22:11.487637Z",
     "shell.execute_reply": "2024-07-18T20:22:11.486291Z"
    },
    "papermill": {
     "duration": 0.385812,
     "end_time": "2024-07-18T20:22:11.491145",
     "exception": false,
     "start_time": "2024-07-18T20:22:11.105333",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing file: /kaggle/input/actions-and-agents-datasets/actions_and_distractors_output.csv\n",
      "CSV file loaded successfully.\n",
      "All required columns are present.\n",
      "Similarity scores computed successfully.\n",
      "Results saved to /kaggle/working/tweets_output_actions_similarity_scores.csv.\n",
      "Results for actions_and_distractors_output.csv saved to tweets_output_actions_similarity_scores.csv:\n",
      "           Target        Cue  Source  List Treated? Distractor1 Distractor2  \\\n",
      "0    baby-clothes     change       4     3        U       scarf      tablet   \n",
      "1    baby-clothes    launder       2     3        U       scarf      tablet   \n",
      "2    baby-clothes      dress       4     3        U       scarf      tablet   \n",
      "3    baby-clothes       wash       1     3        U       scarf      tablet   \n",
      "4    baby-clothes       fold       1     3        U       scarf      tablet   \n",
      "..            ...        ...     ...   ...      ...         ...         ...   \n",
      "754        window   insulate       1     3        U       juice        sink   \n",
      "755        window      cover       1     3        U       juice        sink   \n",
      "756        window  reinforce       1     3        U       juice        sink   \n",
      "757        window       look       1     3        U       juice        sink   \n",
      "758        window       peer       1     3        U       juice        sink   \n",
      "\n",
      "    Distractor3 similarity_cue_target similarity_cue_distractor1  \\\n",
      "0        window              0.209969                   0.040748   \n",
      "1        window              0.210259                   0.193114   \n",
      "2        window              0.468032                   0.550748   \n",
      "3        window              0.417557                   0.239778   \n",
      "4        window              0.274586                   0.253492   \n",
      "..          ...                   ...                        ...   \n",
      "754     flowers               0.29431                    0.10928   \n",
      "755     flowers              0.169399                   0.064345   \n",
      "756     flowers              0.143393                   0.012443   \n",
      "757     flowers              0.120371                   0.117417   \n",
      "758     flowers               0.15497                   0.110263   \n",
      "\n",
      "    similarity_cue_distractor2 similarity_cue_distractor3  \n",
      "0                     0.001815                   0.093764  \n",
      "1                     0.071808                   0.142031  \n",
      "2                     0.106859                   0.176784  \n",
      "3                     0.143488                   0.179183  \n",
      "4                      0.21825                   0.194264  \n",
      "..                         ...                        ...  \n",
      "754                   0.388826                   0.093278  \n",
      "755                   0.151674                   0.172515  \n",
      "756                    0.28689                   0.087577  \n",
      "757                   0.171193                   0.143081  \n",
      "758                   0.030316                   0.136641  \n",
      "\n",
      "[759 rows x 12 columns]\n",
      "Processing file: /kaggle/input/actions-and-agents-datasets/agents_and_distractors_output.csv\n",
      "CSV file loaded successfully.\n",
      "All required columns are present.\n",
      "Similarity scores computed successfully.\n",
      "Results saved to /kaggle/working/tweets_output_agents_similairity_scores.csv.\n",
      "Results for agents_and_distractors_output.csv saved to tweets_output_agents_similairity_scores.csv:\n",
      "           Target        Cue  Source  List Treated? Distractor1 Distractor2  \\\n",
      "0    baby-clothes     infant       4     3        U       scarf      tablet   \n",
      "1    baby-clothes      nanny       2     3        U       scarf      tablet   \n",
      "2    baby-clothes     mother       2     3        U       scarf      tablet   \n",
      "3    baby-clothes     parent       1     3        U       scarf      tablet   \n",
      "4    baby-clothes   retailer       1     3        U       scarf      tablet   \n",
      "..            ...        ...     ...   ...      ...         ...         ...   \n",
      "548        window  decorator       1     3        U       juice        sink   \n",
      "549        window     viewer       1     3        U       juice        sink   \n",
      "550        window   resident       1     3        U       juice        sink   \n",
      "551        window    glazier       1     3        U       juice        sink   \n",
      "552        window   designer       1     3        U       juice        sink   \n",
      "\n",
      "    Distractor3 similarity_cue_target similarity_cue_distractor1  \\\n",
      "0        window              0.493652                   0.307352   \n",
      "1        window              0.282849                   0.101289   \n",
      "2        window              0.345389                   0.135425   \n",
      "3        window              0.249021                   0.038844   \n",
      "4        window              0.171683                    0.15116   \n",
      "..          ...                   ...                        ...   \n",
      "548     flowers              0.239516                   0.103177   \n",
      "549     flowers              0.124194                   0.034962   \n",
      "550     flowers              0.055748                   0.085394   \n",
      "551     flowers              0.227363                   0.070324   \n",
      "552     flowers              0.119248                   0.100216   \n",
      "\n",
      "    similarity_cue_distractor2 similarity_cue_distractor3  \n",
      "0                     0.236176                   0.185571  \n",
      "1                     0.117438                   0.150768  \n",
      "2                     0.132819                    0.16439  \n",
      "3                     0.165623                   0.161303  \n",
      "4                     0.213675                   0.152686  \n",
      "..                         ...                        ...  \n",
      "548                   0.192961                     0.2521  \n",
      "549                   0.057872                  -0.037331  \n",
      "550                   0.085558                   0.017531  \n",
      "551                   0.108344                   0.147446  \n",
      "552                   0.075166                   0.218396  \n",
      "\n",
      "[553 rows x 12 columns]\n"
     ]
    }
   ],
   "source": [
    "# Process each input file individually\n",
    "for input_file_path, output_file_name in zip(input_file_paths, output_file_names):\n",
    "    print(f\"Processing file: {input_file_path}\")\n",
    "    \n",
    "    # Read the CSV file into a DataFrame\n",
    "    df = pd.read_csv(input_file_path)\n",
    "    print(\"CSV file loaded successfully.\")\n",
    "\n",
    "    #Check for presence of 'Action cue' or 'Agent cue' and rename to 'Cue'\n",
    "    if 'Action cue' in df.columns:\n",
    "        df.rename(columns={'Action cue': 'Cue'}, inplace=True)\n",
    "    elif 'Agent cue' in df.columns:\n",
    "        df.rename(columns={'Agent cue': 'Cue'}, inplace=True)\n",
    "    else:\n",
    "        raise ValueError(\"CSV file must contain either 'Action cue' or 'Agent cue' column\")\n",
    "    \n",
    "    # Ensure the DataFrame has the correct columns\n",
    "    required_columns = ['Target', 'Cue', 'Distractor1', 'Distractor2', 'Distractor3']\n",
    "    for column in required_columns:\n",
    "        if column not in df.columns:\n",
    "            raise ValueError(f\"CSV file must contain '{column}' column\")\n",
    "    print(\"All required columns are present.\")\n",
    "    \n",
    "    # Create new columns to store the similarity results\n",
    "    df['similarity_cue_target'] = df.apply(lambda row: compute_similarity(row['Cue'], row['Target']), axis=1)\n",
    "    df['similarity_cue_distractor1'] = df.apply(lambda row: compute_similarity(row['Cue'], row['Distractor1']), axis=1)\n",
    "    df['similarity_cue_distractor2'] = df.apply(lambda row: compute_similarity(row['Cue'], row['Distractor2']), axis=1)\n",
    "    df['similarity_cue_distractor3'] = df.apply(lambda row: compute_similarity(row['Cue'], row['Distractor3']), axis=1)\n",
    "    print(\"Similarity scores computed successfully.\")\n",
    "    \n",
    "    # Save the results to a new CSV file with the desired name\n",
    "    output_csv_file_path = os.path.join('/kaggle/working', output_file_name)  # Update this path if needed\n",
    "    df.to_csv(output_csv_file_path, index=False)\n",
    "    \n",
    "    # Print a sample of the results\n",
    "    print(f\"Results saved to {output_csv_file_path}.\")\n",
    "    print(f\"Results for {os.path.basename(input_file_path)} saved to {output_file_name}:\")\n",
    "    print(df)"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 4517624,
     "sourceId": 7731120,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 5050052,
     "sourceId": 8923607,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30746,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 294.884358,
   "end_time": "2024-07-18T20:22:12.720876",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-07-18T20:17:17.836518",
   "version": "2.5.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
