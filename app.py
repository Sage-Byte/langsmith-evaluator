from quart import Quart, request, jsonify

app = Quart(__name__)
import dotenv


from answer_evaluator import perform_answer_evaluation
from answer_hallucination_evaluator import perform_answer_hallucination_evaluation
from answer_helpfulness_evaluator import perform_answer_helpfulness_evaluation
from docs_relevance_evaluator import perform_docs_relevance_evaluation


dotenv.load_dotenv()


# create index
@app.route('/evaluate', methods=['POST'])
async def create_index():

    data = await request.get_json()
    # api_key = request.headers.get('X-API-Key')
    # if api_key is None:
    #     return jsonify(error='api key is missing missing in the request'), 404    
    dataset_id = data.get('dataset_id')
    if dataset_id is None:
        return jsonify(error='dataset_id  is  missing in the request'), 400
    
    
    perform_answer_evaluation(dataset_id)
    perform_answer_hallucination_evaluation(dataset_id)
    perform_answer_helpfulness_evaluation(dataset_id)
    perform_docs_relevance_evaluation(dataset_id)

    return jsonify(result='check langSmith dashboard for the results'), 200







@app.route('/')
async def hello_world():
    return jsonify(status='app is healthy'), 200

if __name__ == '__main__':
    app.run(debug=True, port=8000)
