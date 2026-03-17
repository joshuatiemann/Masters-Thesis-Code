class ExperimentConfig:
    def __init__(self, experiment, actor, transform, terrain, live_recording=False, video=False, contact_forces=False, total_frames=1_000_000, lr=3e-4, max_grad_norm=1,
                 frames_per_batch=2048, sub_batch_size=64, num_epochs=10,
                 clip_epsilon=0.2, gamma=0.99, lmbda=0.95, entropy_eps=1e-4):
        self.experiment = experiment
        self.actor = actor
        self.transform = transform
        self.terrain = terrain
        self.video = video
        self.contact_forces = contact_forces
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.sub_batch_size = sub_batch_size
        self.num_epochs = num_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.file_path = f"Logs/{experiment}"
        self.live_recording = live_recording

mlp_config = ExperimentConfig(
    experiment = "mlp",
    actor = "mlp_actor",
    transform = "Notransform",
    terrain = "flat",
)

joint_config = ExperimentConfig(
    experiment= "joint",
    actor = "joint_actor",
    transform = "joint_graph",
    terrain = "flat"
)

leg_config = ExperimentConfig(
    experiment='leg',
    actor='leg_actor',
    transform='leg_graph',
    terrain='flat'
)
