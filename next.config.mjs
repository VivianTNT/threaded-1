/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals = [...(config.externals || []), 'canvas', 'pdf-parse']
    }
    // Prevent canvas errors for react-pdf-viewer
    config.resolve.alias = {
      ...config.resolve.alias,
      canvas: false,
    }
    return config
  },
}

export default nextConfig
