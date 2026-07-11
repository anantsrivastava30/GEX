# Multi-stage build for the Next.js frontend.
#
# Uses Next's standalone output so the runtime image ships only server.js +
# the traced node_modules, not the full dependency tree.
#
# BACKEND_URL is baked at BUILD time because Next compiles rewrites into the
# routes manifest. In docker-compose it points at the backend service name;
# to target a different backend, rebuild with --build-arg BACKEND_URL=...

FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-slim AS build
WORKDIR /app
ARG BACKEND_URL=http://backend:8000
ENV BACKEND_URL=$BACKEND_URL
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Run as a non-root user.
RUN groupadd -r nodejs && useradd -r -g nodejs -m nextjs

# Standalone server + assets it does not bundle by default.
COPY --from=build /app/public ./public
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
