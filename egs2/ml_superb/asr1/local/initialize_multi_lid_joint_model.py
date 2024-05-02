import torch
import sys


asr_model = torch.load(sys.argv[1],map_location=torch.device('cpu') )
lid_model = torch.load(sys.argv[2],map_location=torch.device('cpu') )
joint_model = torch.load(sys.argv[3],map_location=torch.device('cpu') )


output_model = sys.argv[4]

def copy_model_parameters(asr_model, lid_model, joint_model):
    asr_state_dict = asr_model
    lid_state_dict = lid_model
    joint_state_dict = joint_model["model"]
    # asr_update_state_dict = {name: param for name, param in asr_state_dict.items() if name in joint_state_dict and param.shape == joint_state_dict[name].shape}
    asr_update_state_dict = {}
    for name, param in asr_state_dict.items():
        name = name.replace("featurizer.", "asr_featurizers.0.")
        if name in joint_state_dict and param.shape == joint_state_dict[name].shape:
            asr_update_state_dict[name] = param
        else:
            print(f"Parameter {name} in ASR model not found in joint model")

    
    # lid_update_state_dict = {name: param for name, param in lid_state_dict.items() if name in joint_state_dict and param.shape == joint_state_dict[name].shape}
    lid_update_state_dict = {}
    for name, param in lid_state_dict.items():
        # name = name.replace("ctc.", "ctc_lid.")
        if "frontend.upstream.upstream.model." not in name:
            name = name.replace("encoder.", "encoder_lid.")
            name = name.replace("projector.", "projector_lid.")
            name = name.replace("loss.", "loss_lid.")
        
        if name in joint_state_dict and param.shape == joint_state_dict[name].shape:
            lid_update_state_dict[name] = param
        elif "featurizer" in name:
            for i in range(25):
                new_name = name.replace("featurizer", f"lid_featurizers.{i}")
                if new_name in joint_state_dict and param.shape == joint_state_dict[new_name].shape:
                    lid_update_state_dict[new_name] = param
        elif "preencoder_lid" in name:
            # copy preencoder parameters to multiple preencoder
            for i in range(25):
                new_name = name.replace("preencoder_lid", f"preencoder_lid.{i}")
                if new_name in joint_state_dict and param.shape == joint_state_dict[new_name].shape:
                    lid_update_state_dict[new_name] = param
        else:
            print(f"Parameter {name} in LID model not found in joint model")

    film_update_state_dict = {}
    for name, param in joint_state_dict.items():
        if "linear_scale.weight" in name:
            film_update_state_dict[name] = torch.zeros_like(param)
        elif "linear_scale.bias" in name:
            film_update_state_dict[name] = torch.ones_like(param)
        elif "linear_shift.weight" in name or "linear_shift.bias" in name:
            film_update_state_dict[name] = torch.zeros_like(param)

    new_joint_state_dict = joint_state_dict.copy()
    new_joint_state_dict.update(lid_update_state_dict)
    new_joint_state_dict.update(asr_update_state_dict)
    new_joint_state_dict.update(film_update_state_dict)


    # new_joint_state_dict = update_joint_lstm_parameters(new_joint_state_dict, asr_state_dict)

    joint_model["model"] = new_joint_state_dict

    unchanged_parameters = []
    changed_parameters = []
    unloaded_asr_parameters = []
    unloaded_lid_parameters = []
    for name, param in joint_state_dict.items():
        if torch.all(torch.eq(param, new_joint_state_dict[name])):
            unchanged_parameters.append(name)
        else:
            changed_parameters.append(name)

    for name, param in asr_state_dict.items():
        if name not in changed_parameters:
            unloaded_asr_parameters.append(name)

    for name, param in lid_state_dict.items():
        if name not in changed_parameters:
            unloaded_lid_parameters.append(name)


    print(f"Unchanged parameters: {unchanged_parameters}")
    # print(f"Unloaded asr parameters: {unloaded_asr_parameters}")
    # print(f"Unloaded lid parameters: {unloaded_lid_parameters}")
    print(f"Changed parameters: {changed_parameters}")


    
    
    torch.save(joint_model, output_model)

    # torch.save(joint_model["model"], output_model.replace("checkpoint", "0epoch"))
    
    same_parameters = []
    different_parameters = []
    other_asr_parameters = []
    other_lid_parameters = []
    for name, param in asr_state_dict.items():
        if name not in lid_state_dict or param.shape != lid_state_dict[name].shape:
            other_asr_parameters.append(name)
        elif torch.all(torch.eq(param, lid_state_dict[name])):
            same_parameters.append(name)
        else:
            different_parameters.append(name)

    for name, param in lid_state_dict.items():
        if name not in same_parameters and name not in different_parameters:
            other_lid_parameters.append(name)

    # print(f"Same parameters for ASR and LID: {same_parameters}")
    print(f"Different parameters for ASR and LID: {different_parameters}")
    # print(f"Other ASR parameters: {other_asr_parameters}")
    # print(f"Other LID parameters: {other_lid_parameters}")


    


unchanged_parameters = copy_model_parameters(asr_model, lid_model, joint_model)