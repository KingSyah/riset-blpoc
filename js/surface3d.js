/* ============================================================
   3D Surface Plot using Three.js
   Renders correlation map as a 3D mesh like MATLAB surf()
   ============================================================ */

const Surface3D = (() => {
  let scene, camera, renderer, mesh, container;
  let isInitialized = false;
  let animId = null;

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
    camera.position.set(1.8, 1.5, 1.8);
    camera.lookAt(0.3, 0, 0.3);

    renderer = new THREE.WebGLRenderer({ canvas: container, antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Lighting
    const ambient = new THREE.AmbientLight(0x404060, 0.6);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(2, 3, 1);
    scene.add(dir);

    // Axes helper (subtle)
    const axesGroup = new THREE.Group();

    // X axis
    const xGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, -0.01, 0),
      new THREE.Vector3(1, -0.01, 0)
    ]);
    const xLine = new THREE.Line(xGeo, new THREE.LineBasicMaterial({ color: 0x666666 }));
    axesGroup.add(xLine);

    // Z axis
    const zGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, -0.01, 0),
      new THREE.Vector3(0, -0.01, 1)
    ]);
    const zLine = new THREE.Line(zGeo, new THREE.LineBasicMaterial({ color: 0x666666 }));
    axesGroup.add(zLine);

    scene.add(axesGroup);

    // Mouse rotation
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    let rotY = Math.PI * 0.25;
    let rotX = 0.5;

    container.addEventListener('mousedown', (e) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    });

    container.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      rotY += dx * 0.005;
      rotX = Math.max(0.1, Math.min(1.2, rotX + dy * 0.005));
      prevMouse = { x: e.clientX, y: e.clientY };
      updateCamera();
    });

    container.addEventListener('mouseup', () => isDragging = false);
    container.addEventListener('mouseleave', () => isDragging = false);

    // Scroll zoom
    container.addEventListener('wheel', (e) => {
      e.preventDefault();
      camera.position.multiplyScalar(e.deltaY > 0 ? 1.05 : 0.95);
    }, { passive: false });

    function updateCamera() {
      const dist = camera.position.length();
      camera.position.x = dist * Math.sin(rotY) * Math.cos(rotX);
      camera.position.y = dist * Math.sin(rotX);
      camera.position.z = dist * Math.cos(rotY) * Math.cos(rotX);
      camera.lookAt(0.3, 0.1, 0.3);
    }

    isInitialized = true;
  }

  /**
   * Render a correlation map as a 3D surface.
   * @param {Float32Array} data  — flat array, rows × cols
   * @param {number} rows        — grid rows
   * @param {number} cols        — grid cols (optional, defaults to rows)
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

    // Create geometry
    const geometry = new THREE.PlaneGeometry(1, 1, gCols - 1, gRows - 1);
    const positions = geometry.attributes.position.array;
    const colors = new Float32Array(positions.length);

    for (let j = 0; j < gRows; j++) {
      for (let i = 0; i < gCols; i++) {
        const idx = j * gCols + i;
        const srcIdx = (j * stepR) * cols + (i * stepC);
        const val = (data[srcIdx] - minV) / range;

        // Position: x = i, y = height, z = j
        positions[idx * 3]     = i / gCols;          // x
        positions[idx * 3 + 1] = val * 0.4;          // y (height)
        positions[idx * 3 + 2] = j / gRows;          // z

        // Color (JET)
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
    mesh.rotation.x = -Math.PI / 2;  // lay flat
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

  return { init, render, resize };
})();
