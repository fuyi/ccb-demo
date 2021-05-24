import json
import pprint
import uuid

from vowpalwabbit import pyvw

vw_options = {
    'quiet': True,
    'ccb_explore_adf': True,
    'epsilon': 0.2,
    'q': 'UA',
    # 'power_t': 0,
    # 'l': 0.05
}
ccb_workspace = pyvw.vw(
    **vw_options
)

print(dir(ccb_workspace))

prediction_history = {}
reward_history = {}
bandit_data = []
DEFAULT_REWARD = 0


def get_options():
    return vw_options


def join_rewards_and_prediction(prediction_id):
    prediction_dict = prediction_history.get(prediction_id)
    if not prediction_dict:
        return None
    rewards = reward_history.get(prediction_id)
    action_indices = prediction_dict['actionIndices']
    slot_components = prediction_dict['components'][-len(action_indices):]
    context_components = prediction_dict['components'][:len(prediction_dict['components'])-len(slot_components)]
    for k in range(0, len(action_indices)):
        slot_string = slot_components[k]
        slot_feature_string = slot_string.split('|')[1]
        if rewards and rewards.get(k):
            v = rewards.get(k)
            slot_string_with_label = f"ccb slot {action_indices[k]}:{int(v)}:{prediction_dict['probabilities'][k]} |{slot_feature_string}"
        else:
            slot_string_with_label = f"ccb slot {action_indices[k]}:{DEFAULT_REWARD}:{prediction_dict['probabilities'][k]} |{slot_feature_string}"
        slot_components[k] = slot_string_with_label
        
    return '\n'.join(context_components + slot_components)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def create_shared_user_feature_string_list(user_feature_dict):
    feature_list = []
    for k, v in user_feature_dict.items():
        if is_number(v):
            feature_list.append(f"{k}:{v}")
        else:
            feature_list.append(f"{k}_{v}")
    return [f"ccb shared |User {' '.join(feature_list)}"]


def create_action_feature_string_list(action_feature_dicts):
    action_feature_string_list = []
    for action_dict in action_feature_dicts:
        feature_list = []
        for k, v in action_dict.items():
            if is_number(v):
                feature_list.append(f"{k}:{v}")
            else:
                feature_list.append(f"{k}_{v}")
        action_feature_string_list.append(f"ccb action |Action {' '.join(feature_list)}")
    return action_feature_string_list


def create_slots_string_list(slot_count):
    slot_string_list = []
    for i in range(0, slot_count):
        slot_string_list.append('ccb slot |Slot')
    return slot_string_list


def create_prediction_example_string(json_dict):
    example_string_components = []
    example_string_components += create_shared_user_feature_string_list(json_dict.get('userFeatures'))
    example_string_components += create_action_feature_string_list(json_dict.get('actionFeatures'))
    example_string_components += create_slots_string_list(json_dict.get('slotCount'))
    return '\n'.join(example_string_components), example_string_components


def ccb_predict(data):
    ccb_ex, ex_components = create_prediction_example_string(data)
    prediction_pdf_list = ccb_workspace.predict(ccb_ex)
    pprint.pprint(prediction_pdf_list)
    prediction_tuple_list = [prediction_pdf[0] for prediction_pdf in prediction_pdf_list]
    print(prediction_tuple_list)
    recommendations_list = []
    prediction_id = str(uuid.uuid4())
    for prediction_tuple in prediction_tuple_list:
        action_index, probability = prediction_tuple
        recommendations_list.append(
            {
                'actionIndex': action_index,
                'probability': round(probability, 6)
            }
        )
    prediction_history[prediction_id] = {
        'components': ex_components,
        'actionIndices': [r['actionIndex'] for r in recommendations_list],
        'probabilities': [r['probability'] for r in recommendations_list]
    }
    return prediction_id, recommendations_list


def ccb_learn(prediction_id):
    ccb_ex = join_rewards_and_prediction(prediction_id)
    if ccb_ex:
        ccb_workspace.learn(ccb_ex)
        bandit_data.append(ccb_ex)
    else:
        print('SKIPPED')


def ccb_save_reward(predictionId, slotId, reward):
    if predictionId not in reward_history:
        reward_history[predictionId] = {}
    reward_history[predictionId][slotId] = -reward


def save_bandit_data_to_disk():
    global bandit_data
    with open('ccb-bandit-data.txt', 'a') as f:
        for line in bandit_data:
            f.write(line + '\n\n')
    bandit_data = []


if __name__ == "__main__":
    with open('example.json') as f:
        data = json.load(f)
    for i in range(0, 1000):
        prediction_id, recommendations_list = ccb_predict(data)
        print('Prediction:')
        pprint.pprint(recommendations_list)
        print('Sending rewards...')
        ccb_save_reward(prediction_id, 0, 10)
        print('Joining & learning...')
        ccb_learn(prediction_id)
        print('Saving data to disk...')
        save_bandit_data_to_disk()
