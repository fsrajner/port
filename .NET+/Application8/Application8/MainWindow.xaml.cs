using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Application8
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            TranslateTransform3D tt = new TranslateTransform3D();
            tt.OffsetX = tt.OffsetY = tt.OffsetZ = e.Delta / 100;
            (this.camera.Transform as MatrixTransform3D).Matrix *= tt.Value;
        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            RotateTransform3D spin = null;
            int mul = 1;
            if (e.Key == Key.A || e.Key == Key.W) mul = -1;
            if (e.Key == Key.A || e.Key == Key.D)
            {
                spin = new RotateTransform3D(new AxisAngleRotation3D(new Vector3D(0, 0, 1), mul * 5));
            }
            if (e.Key == Key.W || e.Key == Key.S)
            {
                spin = new RotateTransform3D(new AxisAngleRotation3D(new Vector3D(0, 1, 0), mul * 5));
            }
            spin.CenterX = spin.CenterY = spin.CenterZ = 5;

            (this.cube.Transform as MatrixTransform3D).Matrix *= spin.Value;
        }

        private void Start_Click(object sender, RoutedEventArgs e)
        {
            var me = (MediaElement)(grid.Resources["panel"] as DockPanel).Children[0];

            me.Play();
        }

        private void Stop_Click(object sender, RoutedEventArgs e)
        {
            var me = (MediaElement)(grid.Resources["panel"] as DockPanel).Children[0];

            me.Stop();
        }
    }
}
