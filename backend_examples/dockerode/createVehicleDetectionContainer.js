const Docker = require("dockerode");

const docker = new Docker();

function assertRequiredEnv(env) {
  const required = ["STREAM_PORT"];
  for (const key of required) {
    if (env[key] === undefined || env[key] === null || String(env[key]).trim() === "") {
      throw new Error(`Missing required env: ${key}`);
    }
  }

  if (!env.TRACK_RTSP_URL && !env.RTSP_URL) {
    throw new Error("Either TRACK_RTSP_URL or RTSP_URL is required");
  }
}

function toEnvArray(env) {
  return Object.entries(env).map(([key, value]) => `${key}=${String(value)}`);
}

function buildPortBindings(env, explicitPortBindings = {}) {
  const containerPort = String(env.STREAM_PORT || "8005");
  const hostPort = explicitPortBindings[containerPort]?.hostPort || containerPort;

  return {
    exposedPorts: {
      [`${containerPort}/tcp`]: {},
    },
    portBindings: {
      [`${containerPort}/tcp`]: [{ HostPort: String(hostPort) }],
    },
  };
}

function buildVolumeBinds(volumeBindings = []) {
  return volumeBindings.map((item) => {
    if (typeof item === "string") return item;
    return `${item.hostPath}:${item.containerPath}${item.readOnly ? ":ro" : ""}`;
  });
}

function buildGpuHostConfig(profile) {
  if (!profile.gpu_enabled || profile.gpu_vendor === "none") {
    return {};
  }

  if (profile.gpu_vendor === "nvidia") {
    return {
      DeviceRequests: [
        {
          Driver: "nvidia",
          Count: profile.gpu_count && profile.gpu_count > 0 ? profile.gpu_count : -1,
          Capabilities: [["gpu", "compute", "utility", "video"]],
        },
      ],
    };
  }

  if (profile.gpu_vendor === "intel") {
    return {
      Devices: [
        {
          PathOnHost: "/dev/dri",
          PathInContainer: "/dev/dri",
          CgroupPermissions: "rwm",
        },
      ],
    };
  }

  throw new Error(`Unsupported gpu_vendor: ${profile.gpu_vendor}`);
}

async function createVehicleDetectionContainer(profile) {
  const env = profile.env_json || {};
  assertRequiredEnv(env);

  const { exposedPorts, portBindings } = buildPortBindings(
    env,
    profile.port_bindings_json || {}
  );

  const hostConfig = {
    Binds: buildVolumeBinds(profile.volume_bindings_json || []),
    PortBindings: portBindings,
    RestartPolicy: {
      Name: profile.restart_policy_name || "unless-stopped",
      MaximumRetryCount: profile.restart_policy_max_retry_count || 0,
    },
    NetworkMode: profile.network_mode || "bridge",
    ...buildGpuHostConfig(profile),
  };

  const envForDocker = { ...env };

  if (profile.gpu_enabled && profile.gpu_vendor === "nvidia") {
    envForDocker.NVIDIA_VISIBLE_DEVICES =
      envForDocker.NVIDIA_VISIBLE_DEVICES || "all";
    envForDocker.NVIDIA_DRIVER_CAPABILITIES =
      envForDocker.NVIDIA_DRIVER_CAPABILITIES || "compute,utility,video";
  }

  const container = await docker.createContainer({
    name: profile.container_name,
    Image: profile.image,
    Env: toEnvArray(envForDocker),
    ExposedPorts: exposedPorts,
    HostConfig: hostConfig,
    Labels: {
      "app.service": "vehicle-detection",
      "app.profile-id": String(profile.id),
    },
  });

  return container;
}

async function createAndStartVehicleDetectionContainer(profile) {
  const container = await createVehicleDetectionContainer(profile);
  await container.start();
  return container;
}

module.exports = {
  createVehicleDetectionContainer,
  createAndStartVehicleDetectionContainer,
};
