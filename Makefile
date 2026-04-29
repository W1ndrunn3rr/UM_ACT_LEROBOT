PYTHON_BIN := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
EXPERIMENTS := baseline baseline_no_vae canny canny_no_vae grayscale sobel blur_canny downsample_84

FOLLOWER_PORT ?= /dev/tty.usbmodem5AAF2630541
LEADER_PORT ?= /dev/tty.usbmodem5AAF2634711
FOLLOWER_ID ?= blue_follower
LEADER_ID ?= blue_leader_2
DATASET_ID ?= W1ndrunn3rr/pick_and_lift_v2
POLICY_ID ?= W1ndrunn3rr/act_pick_and_lift_v2
DATASET_ROOT ?= workspace/pick_and_lift_v2

EXP ?= baseline
MODEL ?= $(POLICY_ID)
FILTER ?= none
ROBOT_TYPE ?= so101_follower
ROBOT_PORT ?= $(FOLLOWER_PORT)
ROBOT_ID ?= $(FOLLOWER_ID)
CAMERA_INDEX ?= 0
CAMERA_WIDTH ?= 1280
CAMERA_HEIGHT ?= 720
CAMERA_FPS ?= 30
CAMERA_BACKEND ?= 1200
EVAL_DATASET_REPO ?= W1ndrunn3rr/eval_pick_and_lift_v2
EVAL_EPISODES ?= 3
EPISODE_TIME ?= 180
RESET_TIME ?= 180
TASK ?= pick_and_lift

.PHONY: help train train-all eval

help:
	@echo "make train EXP=baseline"
	@echo "make train-all"
	@echo "make eval FILTER=canny"
	@echo "Defaults: MODEL=$(MODEL) ROBOT_PORT=$(ROBOT_PORT) ROBOT_ID=$(ROBOT_ID)"

train:
	$(PYTHON_BIN) -m src.scripts.train $(EXP)

train-all:
	@for exp in $(EXPERIMENTS); do \
		echo "==> $$exp"; \
		$(PYTHON_BIN) -m src.scripts.train $$exp || exit $$?; \
	done

eval:
	@test -n "$(MODEL)" || (echo "MODEL is required, e.g. make eval MODEL=user/policy FILTER=canny" && exit 1)
	$(PYTHON_BIN) -m src.scripts.lerobot_record_filtered \
		--robot.type=$(ROBOT_TYPE) \
		--robot.port=$(ROBOT_PORT) \
		--robot.id=$(ROBOT_ID) \
		--robot.cameras='{front: {type: filtered_opencv, index_or_path: $(CAMERA_INDEX), width: $(CAMERA_WIDTH), height: $(CAMERA_HEIGHT), fps: $(CAMERA_FPS), backend: $(CAMERA_BACKEND), filter_name: $(FILTER)}}' \
		--display_data=true \
		--dataset.repo_id=$(EVAL_DATASET_REPO) \
		--dataset.num_episodes=$(EVAL_EPISODES) \
		--dataset.episode_time_s=$(EPISODE_TIME) \
		--dataset.reset_time_s=$(RESET_TIME) \
		--dataset.single_task=$(TASK) \
		--dataset.push_to_hub=false \
		--policy.path=$(MODEL)
