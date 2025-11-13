from conf.conf import Config

def calculate_reward(step_info, next_step_info):
    '''
    Calculate the reward for a given step based on the step information.
    '''
    reward = 0 
    prev_player = step_info['players'][0]
    next_player = next_step_info['players'][0]

    reward_info = Config.reward_info.copy()

    # looking at node reward
    if next_player["nodeInView"] == 1:
        reward += 0.05
        reward_info["looking_at_node"] = 0.05

    # resource reward
    prev_resources = 0
    for item in prev_player['items']:
        if item["displayName"] == "Stone" or item["displayName"] == "Metal Ore" or item["displayName"] == "Sulfur Ore":
            prev_resources += item['amount']
    next_resources = 0
    for item in next_player['items']:
        if item["displayName"] == "Stone" or item["displayName"] == "Metal Ore" or item["displayName"] == "Sulfur Ore":
            next_resources += item['amount']

    if next_resources > prev_resources:
        reward += 4
        reward_info["resource_gathered"] = 2

    # closest node reward
    prev_closest_dist = float('inf')
    next_closest_dist = float('inf')
    for node in step_info["nodes"]: 
        node_pos = node["position"]
        dist = ((node_pos["x"] - prev_player["position"]["x"]) ** 2 +
            (node_pos["y"] - prev_player["position"]["y"]) ** 2 +
            (node_pos["z"] - prev_player["position"]["z"]) ** 2) ** 0.5
        if dist < prev_closest_dist:
            prev_closest_dist = dist
    for node in next_step_info["nodes"]: 
        node_pos = node["position"]
        dist = ((node_pos["x"] - next_player["position"]["x"]) ** 2 +
            (node_pos["y"] - next_player["position"]["y"]) ** 2 +
            (node_pos["z"] - next_player["position"]["z"]) ** 2) ** 0.5
        if dist < next_closest_dist:
            next_closest_dist = dist
            
    # we only bother with closest node rwd if node is within 20 units and visible
    if next_closest_dist < 20.0 and next_player["nodeInView"] == 1:
        if next_closest_dist < prev_closest_dist:
            reward += 0.1
            reward_info["closest_node"] = 0.1
        elif next_closest_dist > prev_closest_dist:
            reward -= 0.1
            reward_info["closest_node"] = -0.1

    # swimming penalty
    if next_player["isSwimming"]:
        reward -= 0.1
        reward_info["swimming_penalty"] = -0.1

    return reward, reward_info



"""
example info:
{
      "prefab": "metal-ore",
      "position": {
        "x": -404.465637,
        "y": -49.72604,
        "z": 82.32504
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 458.7896,
        "y": -49.9908333,
        "z": -343.2428
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 316.617737,
        "y": -49.9908333,
        "z": -441.896881
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -388.704651,
        "y": -49.257843,
        "z": 124.898849
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -442.405548,
        "y": -49.9908333,
        "z": -483.432373
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 139.288177,
        "y": -5.09616756,
        "z": -165.2415
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 480.595154,
        "y": -49.9908333,
        "z": -190.897476
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 79.29663,
        "y": 5.744636,
        "z": -57.0397758
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -249.552933,
        "y": -32.50906,
        "z": 246.243561
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -413.950439,
        "y": -49.9908333,
        "z": 385.3618
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 206.063721,
        "y": 3.16739082,
        "z": 108.038826
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -471.449432,
        "y": -49.9908333,
        "z": -444.9637
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -329.750732,
        "y": -32.1616821,
        "z": -247.34111
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -66.74277,
        "y": -0.448316336,
        "z": 94.2247849
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 139.856781,
        "y": 11.8250256,
        "z": 121.1422
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 499.351746,
        "y": -49.9908333,
        "z": 348.8208
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 248.043259,
        "y": -49.32624,
        "z": 414.280762
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -156.002136,
        "y": -46.2647667,
        "z": 379.9721
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -21.956686,
        "y": -1.79386139,
        "z": 16.23772
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -415.776184,
        "y": -49.9908333,
        "z": 208.342529
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 26.9480343,
        "y": -1.96394324,
        "z": 230.568192
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 276.131042,
        "y": -5.448431,
        "z": -94.76836
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -332.485382,
        "y": -49.9908333,
        "z": -470.956116
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 444.855072,
        "y": -49.8849754,
        "z": -30.9611073
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 229.9516,
        "y": 1.865983,
        "z": 47.3379173
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -37.133934,
        "y": -49.9908333,
        "z": 439.43985
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -267.34964,
        "y": -18.6334839,
        "z": -248.904678
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -200.026352,
        "y": -10.6118917,
        "z": -161.993713
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -428.0044,
        "y": -49.9908333,
        "z": -137.934143
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -39.2556229,
        "y": -0.979989767,
        "z": 50.50726
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 377.431671,
        "y": -49.9908333,
        "z": -413.974548
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 210.400925,
        "y": -49.07805,
        "z": 406.3092
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 395.3745,
        "y": -49.4408,
        "z": 363.725769
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -404.4513,
        "y": -49.9908333,
        "z": 331.837128
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 188.312637,
        "y": -33.7629,
        "z": 303.188171
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 124.65345,
        "y": 10.1698036,
        "z": 114.046272
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -459.7872,
        "y": -49.9908333,
        "z": -484.822754
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -182.4532,
        "y": -41.7172623,
        "z": 351.573456
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 13.0909081,
        "y": -33.43922,
        "z": -273.0256
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -298.3397,
        "y": -43.58566,
        "z": 204.135437
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -211.2067,
        "y": -23.6677532,
        "z": 206.770233
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -77.26722,
        "y": -49.9908333,
        "z": -432.5766
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 37.7832642,
        "y": -49.9908333,
        "z": -480.449341
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 351.22113,
        "y": -29.0723743,
        "z": 67.4915543
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 105.907471,
        "y": -48.89959,
        "z": -362.898346
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -422.3542,
        "y": -49.9908333,
        "z": 111.661018
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 25.8391418,
        "y": -2.22229958,
        "z": -52.65786
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -104.567459,
        "y": -17.262846,
        "z": -79.46844
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 462.000671,
        "y": -49.91588,
        "z": 59.8848877
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -412.930878,
        "y": -49.9908333,
        "z": -302.689972
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -28.9053059,
        "y": -0.13205409,
        "z": 193.714218
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -327.255219,
        "y": -45.7024879,
        "z": 306.7554
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -158.759674,
        "y": -17.0317879,
        "z": -234.420151
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -437.771545,
        "y": -49.9908333,
        "z": -413.129822
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 191.605469,
        "y": -44.8825951,
        "z": -311.008331
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 1.08378267,
        "y": -0.122189522,
        "z": 211.639328
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -462.010956,
        "y": -49.9908333,
        "z": -64.55491
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 357.7438,
        "y": -35.23582,
        "z": 175.821152
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -204.8538,
        "y": -49.19046,
        "z": 393.360779
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 443.2066,
        "y": -49.83133,
        "z": -39.2679024
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -351.4386,
        "y": -36.93238,
        "z": -49.28587
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 461.6939,
        "y": -49.9908333,
        "z": -119.87355
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 416.123779,
        "y": -47.27617,
        "z": -66.8373
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -205.8087,
        "y": -22.8970642,
        "z": 236.4055
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 299.0533,
        "y": -8.768856,
        "z": -36.3893127
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -346.476,
        "y": -47.6509323,
        "z": 203.827072
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -412.355682,
        "y": -49.399704,
        "z": -39.0830956
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 19.9802246,
        "y": -49.9908333,
        "z": -450.9199
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -435.978546,
        "y": -49.9908333,
        "z": 113.3509
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 402.797028,
        "y": -49.9908333,
        "z": 377.598572
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 338.828217,
        "y": -23.3337879,
        "z": -101.966278
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": -271.7737,
        "y": -49.9908333,
        "z": 413.667053
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": 272.254974,
        "y": -20.1134682,
        "z": 229.911575
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 73.9083557,
        "y": 14.2660141,
        "z": 84.96001
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -115.874481,
        "y": -10.0076494,
        "z": 243.797745
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": -230.7678,
        "y": -49.9908333,
        "z": -441.3095
      },
      "health": 500.0
    },
    {
      "prefab": "stone-ore",
      "position": {
        "x": 396.32135,
        "y": -49.9908333,
        "z": -358.377777
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -445.3012,
        "y": -49.9908333,
        "z": 355.5623
      },
      "health": 500.0
    },
    {
      "prefab": "metal-ore",
      "position": {
        "x": 159.292938,
        "y": 8.059621,
        "z": 128.581192
      },
      "health": 500.0
    },
    {
      "prefab": "sulfur-ore",
      "position": {
        "x": -439.6243,
        "y": -49.9908333,
        "z": 397.672455
      },
      "health": 500.0
    },
    {
      "prefab": "ore_stone",
      "position": {
        "x": 152.002853,
        "y": -7.52491474,
        "z": 113.575089
      },
      "health": 500.0
    },
    {
      "prefab": "ore_sulfur",
      "position": {
        "x": 91.2328339,
        "y": -16.3099136,
        "z": 113.745125
      },
      "health": 500.0
    },
    {
      "prefab": "ore_metal",
      "position": {
        "x": 97.4827957,
        "y": -37.3119125,
        "z": 94.18051
      },
      "health": 500.0
    },
    {
      "prefab": "ore_stone",
      "position": {
        "x": 78.77614,
        "y": 1.73608541,
        "z": 73.91438
      },
      "health": 500.0
    }
  ],
  "playersCount": 0,
  "nodesCount": 312
}
"""