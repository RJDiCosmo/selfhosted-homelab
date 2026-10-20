# ACE-Step UI (fspecii/ace-step-ui), pinned commit, built as: nginx (static
# frontend + reverse proxy to the API) + Express backend in one container.
FROM node:22-bookworm-slim
ARG UI_REF=a1fdf91829ec6f7b98844f80e323529cd155dbf2
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates nginx ffmpeg python3 make g++ curl \
    && rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default
WORKDIR /app
RUN git clone https://github.com/fspecii/ace-step-ui.git . && git checkout "$UI_REF"
COPY patch-gradio-args.py patch-auth-header.py /tmp/
RUN python3 /tmp/patch-gradio-args.py && python3 /tmp/patch-auth-header.py
RUN npm ci && npm run build
RUN cd server && npm ci && mkdir -p public/audio
COPY nginx.conf /etc/nginx/conf.d/studio.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV NODE_ENV=production PORT=3001
EXPOSE 3000
CMD ["/entrypoint.sh"]
