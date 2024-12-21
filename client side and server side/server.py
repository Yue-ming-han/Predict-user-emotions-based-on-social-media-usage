from flask_cors import CORS  # 导入CORS
from flask import Flask, request, jsonify
import joblib
import pandas as pd

GaussianNB_model = joblib.load(r'D:\social-media-usage-and-emotional-well-being\pythonProject\models\bagging_clf.pickle')
features = ['Age', 'Gender', 'Platform', 'Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day','Comments_Received_Per_Day', 'Messages_Sent_Per_Day']
test_x = pd.read_csv(r"D:\social-media-usage-and-emotional-well-being\pythonProject\Data preprocessing\test_x.csv")
app = Flask(__name__)
CORS(app)


@app.route('/api/submit', methods=['POST'])
def handle_submit():
    data = request.get_json()
    user_data = data.get('userData')
    d = pd.DataFrame([user_data], columns=features)
    if d.loc[0, "Gender"] == 'Male':
        d["Gender_Male"] = [1]
    else:
        d["Gender_Non-binary"] = [1]
    d["Platform" + "_" + str(d.loc[0, "Platform"])] = [1]
    d.drop("Platform", axis=1)
    d.drop("Gender", axis=1)

    missing_cols_test = set(test_x.columns) - set(d.columns)
    for col in missing_cols_test:
        d[col] = 0
    d = d[test_x.columns]

    print(f"接收到的用户数据: {d}")
    pred = GaussianNB_model.predict(d)[0]
    # 在这里可以对接收的数据进行进一步处理，比如调用情绪分析模型等相关业务逻辑
    response_data = {'message': pred}
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
