// BanglaLPR Mobile service worker: keeps the app shell for quick start; plate reading always goes to the PC server.
const SHELL = 'banglalpr-shell-v1';
const FILES = ['/app', '/static/mobile.js', '/manifest.webmanifest', '/static/icon-192.png', '/static/icon-512.png'];
self.addEventListener('install', e => e.waitUntil(caches.open(SHELL).then(c => c.addAll(FILES)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.pathname.startsWith('/api/') || u.pathname.startsWith('/media/')) return;
  if (!FILES.includes(u.pathname)) return;
  // network first (so updates show at once), cached shell when the server is unreachable
  e.respondWith(fetch(e.request).then(r => { caches.open(SHELL).then(c => c.put(e.request, r.clone())); return r; })
    .catch(() => caches.match(e.request)));
});
