/* ============================================================
   3D Surface Plot using Three.js
   Renders correlation map as 3D mesh like MATLAB surf()
   ============================================================ */

const Surface3D = (() => {
  let scene, camera, renderer, mesh, container;
  let isInitialized = false;
  let animId = null;

  // Orbit state (module-level so resetView can access)
  const INITIAL_ROT_Y = Math.PI * 0.75;
  const INITIAL_ROT_X = 0.55;
  const CENTER = new THREE.Vector3(0.5, 0.1, 0.5);
  let rotY = INITIAL_ROT_Y;
  let rotX = INITIAL_ROT_X;
  let orbitDist = 2.598; // sqrt(1.5²+1.5²+1.5²)

  function updateCamera() {
    if (!camera) return;
    camera.position.x = CENTER.x + orbitDist * Math.sin(rotY) * Math.cos(rotX);
    camera.position.y = CENTER.y + orbitDist * Math.sin(rotX);
    camera.position.z = CENTER.z + orbitDist * Math.cos(rotY) * Math.cos(rotX);
    camera.lookAt(CENTER);
  }

  function resetView() {
    rotY = INITIAL_ROT_Y;
    rotX = INITIAL_ROT_X;
    orbitDist = 2.598;
    updateCamera();
  }

  /** JET colormap: value 0-1 → [r,g,b] 0-255 */
  function jet(v) {
    v = Math.max(0, Math.min(1, v));
    let r, g, b;
    if (v < 0.125) {
      r = 0; g = 0; b = 0.5 + v * 4;
    } else if (v < 0.375) {
      r = 0; g = (v - 0.125) * 4; b = 1;
    } else if (v < 0.625) {
      r = (v - 0.375) * 4; g = 1; b = 1 - (v - 0.375) * 4;
    } else if (v < 0.875) {
      r = 1; g = 1 - (v - 0.625) * 4; b = 0;
    } else {
      r = 1 - (v - 0.875) * 4; g = 0; b = 0;
    }
    return [(r * 255) | 0, (g * 255) | 0, (b * 255) | 0];
  }

  function init(canvasEl) {
    if (isInitialized) return;

    container = canvasEl;
    const w = container.clientWidth || 512;
    const h = container.clientHeight || 300;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e17);

    camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
    orbitDist = 2.598;
    updateCamera();

    renderer = new THREE.WebGLRenderer({ canvas: container, antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Lighting
    const ambient = new THREE.AmbientLight(0x404060, 0.6);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(2, 3, 1);
    scene.add(dir);

    // Axes helper
    const axesGroup = new THREE.Group();

    const xGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.5, -0.01, -0.5),
      new THREE.Vector3( 0.5, -0.01, -0.5)
    ]);
    axesGroup.add(new THREE.Line(xGeo, new THREE.LineBasicMaterial({ color: 0x666666 })));

    const zGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.5, -0.01, -0.5),
      new THREE.Vector3(-0.5, -0.01,  0.5)
    ]);
    axesGroup.add(new THREE.Line(zGeo, new THREE.LineBasicMaterial({ color: 0x666666 })));

    scene.add(axesGroup);

    // ── Mouse drag ──
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };

    container.addEventListener('mousedown', (e) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
      container.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      rotY -= dx * 0.005;
      rotX = Math.max(0.1, Math.min(1.2, rotX + dy * 0.005));
      prevMouse = { x: e.clientX, y: e.clientY };
      updateCamera();
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
      container.style.cursor = 'grab';
    });

    // ── Touch drag (mobile) ──
    let touchId = null;
    let prevTouch = { x: 0, y: 0 };

    container.addEventListener('touchstart', (e) => {
      if (e.touches.length !== 1) return;
      const t = e.touches[0];
      touchId = t.identifier;
      prevTouch = { x: t.clientX, y: t.clientY };
    }, { passive: true });

    container.addEventListener('touchmove', (e) => {
      if (touchId === null) return;
      for (const t of e.changedTouches) {
        if (t.identifier !== touchId) continue;
        const dx = t.clientX - prevTouch.x;
        const dy = t.clientY - prevTouch.y;
        rotY -= dx * 0.005;
        rotX = Math.max(0.1, Math.min(1.2, rotX + dy * 0.005));
        prevTouch = { x: t.clientX, y: t.clientY };
        updateCamera();
      }
    }, { passive: true });

    container.addEventListener('touchend', (e) => {
      for (const t of e.changedTouches) {
        if (t.identifier === touchId) { touchId = null; break; }
      }
    }, { passive: true });

    // ── Scroll zoom ──
    container.addEventListener('wheel', (e) => {
      e.preventDefault();
      orbitDist = Math.max(0.5, Math.min(5, orbitDist * (e.deltaY > 0 ? 1.08 : 0.92)));
      updateCamera();
    }, { passive: false });

    isInitialized = true;
  }

  /**
   * Render correlation map as 3D surface.
   * @param {Float32Array} data — flat array, rows × cols
   * @param {number} rows
   * @param {number} cols (optional, defaults to rows)
   */
  function render(data, rows, cols) {
    if (!isInitialized) return;
    cols = cols || rows;

    // Remove old mesh
    if (mesh) {
      scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }

    // Downsample for performance (max 128×128)
    const stepR = Math.max(1, Math.floor(rows / 128));
    const stepC = Math.max(1, Math.floor(cols / 128));
    const gRows = Math.floor(rows / stepR);
    const gCols = Math.floor(cols / stepC);

    // Find min/max
    let minV = Infinity, maxV = -Infinity;
    for (let i = 0; i < rows * cols; i++) {
      if (data[i] < minV) minV = data[i];
      if (data[i] > maxV) maxV = data[i];
    }
    const range = maxV - minV || 1;

    // Create geometry — horizontal, centered at origin
    const geometry = new THREE.PlaneGeometry(1, 1, gCols - 1, gRows - 1);
    const positions = geometry.attributes.position.array;
    const colors = new Float32Array(positions.length);

    for (let j = 0; j < gRows; j++) {
      for (let i = 0; i < gCols; i++) {
        const idx = j * gCols + i;
        const srcIdx = (j * stepR) * cols + (i * stepC);
        const val = (data[srcIdx] - minV) / range;

        positions[idx * 3]     = (i / gCols) - 0.5;
        positions[idx * 3 + 1] = val * 0.4;
        positions[idx * 3 + 2] = (j / gRows) - 0.5;

        const [r, g, b] = jet(val);
        colors[idx * 3]     = r / 255;
        colors[idx * 3 + 1] = g / 255;
        colors[idx * 3 + 2] = b / 255;
      }
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 30,
      specular: 0x222222
    });

    mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(0.5, 0, 0.5);
    scene.add(mesh);

    renderLoop();
  }

  function renderLoop() {
    if (animId) cancelAnimationFrame(animId);
    function animate() {
      animId = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }
    animate();
  }

  function resize() {
    if (!isInitialized || !container) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  return { init, render, resize, resetView };
})();
