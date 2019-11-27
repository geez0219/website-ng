import { Component, OnInit, HostListener, ViewChild, ElementRef } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { ActivatedRoute, Router, UrlSegment } from '@angular/router';
import { BehaviorSubject } from 'rxjs';

import { NestedTreeControl } from '@angular/cdk/tree';
import { MatTreeNestedDataSource, MatTreeModule, MatTreeNode } from '@angular/material/tree';

import { Example } from '../example';
import { MatSidenav} from '@angular/material';
import { MatCardModule } from '@angular/material/card';

import { Title } from '@angular/platform-browser';

@Component({
  selector: 'app-example',
  templateUrl: './example.component.html',
  styleUrls: ['../api/api.component.css']
})
export class ExampleComponent implements OnInit {
  exampleList: Example[];
  selectedExample: string;
  currentSelection: string;
  currentExampleText: string;

  segments: UrlSegment[];

  treeControl: NestedTreeControl<Example>;
  dataSource: MatTreeNestedDataSource<Example>;
  minWidth: number = 640;
  screenWidth: number;
  private screenWidth$ = new BehaviorSubject<number>(window.innerWidth);

  structureHeaderDict = {
    'Content-Type': 'application/json',
    'Accept': "application/json, text/plain",
    'Access-Control-Allow-Origin': '*'
  }
  structureRequestOptions = {
    headers: new HttpHeaders(this.structureHeaderDict),
  };

  contentHeaderDict = {
    'Accept': "application/json, text/plain",
    'Access-Control-Allow-Origin': '*'
  }
  contentRequestOptions = {
    responseType: 'text' as 'text',
    headers: new HttpHeaders(this.contentHeaderDict)
  };

  @ViewChild('sidenav', { static: true })
  sidenav: MatSidenav;

  @ViewChild('grippy', { static: true })
  grippy: ElementRef;

  @HostListener('window:resize', ['$event'])
  onResize(event) {
    this.screenWidth$.next(event.target.innerWidth);
  }

  constructor(private http: HttpClient,
              private router: Router,
              private route: ActivatedRoute,
              private title: Title) { }

  ngOnInit() {
    this.treeControl = new NestedTreeControl<Example>(node => node.children);
    this.dataSource = new MatTreeNestedDataSource<Example>();

    this.route.url.subscribe((segments: UrlSegment[]) => {
      this.segments = segments;
      this.getExampleStructure();
    });

    this.screenWidth$.subscribe(width => {
      this.screenWidth = width;
    });
  }

  hasChild = (_: number, node: Example) => !!node.children && node.children.length > 0;

  flatten(arr) {
    var ret: Example[] = [];
    for (let a of arr) {
      if (a.children) {
        ret = ret.concat(this.flatten(a.children));
      } else {
        ret = ret.concat(a);
      }
    }

    return ret;
  }

  expandNodes(exampleName: string) {
    var exampleParts: Array<string> = exampleName.split("/");
    exampleParts.pop();

    var expandNode = this.exampleList.filter(example => example.name === exampleParts[0])[0];
    this.treeControl.expand(expandNode);
  }

  getExampleStructure() {
    if (this.exampleList) {
      this.loadSelectedExample();
    } else {
      this.http.get('assets/example/structure.json', this.structureRequestOptions).subscribe(data => {
        this.exampleList = <Example[]>(data);

        this.dataSource.data = this.exampleList;
        this.treeControl.dataNodes = this.exampleList;

        this.loadSelectedExample();
      },
      error => {
        console.error(error);
        this.router.navigate(['PageNotFound'], {replaceUrl:true})
      });
    }
  }

  private loadSelectedExample() {
    if (this.segments.length == 0) {
      this.updateExampleContent(this.exampleList[0].children[0]);
      this.treeControl.expand(this.treeControl.dataNodes[0]);
    }
    else {
      var e: Example[] = this.flatten(this.exampleList)
        .filter(example =>
          (this.segments.map(segment => segment.toString()).join('/') + ".md") === example.name);

      if (e.length > 0) {
        this.updateExampleContent(e[0]);
        this.expandNodes(e[0].name);
      } else {
        this.router.navigate(['PageNotFound'], {replaceUrl:true});
      }
    }
  }

  private updateExampleContent(example: Example) {
    window.scroll(0, 0);

    this.selectedExample = example.name;
    this.currentSelection = 'assets/example/' + example.name;

    this.getSelectedExampleText();
    this.title.setTitle(example.displayName + " | Fastestimator");
  }

  getSelectedExampleText() {
    this.http.get(this.currentSelection, this.contentRequestOptions).subscribe(data => {
      this.currentExampleText = data;
    },
    error => {
      console.error(error);
      this.router.navigate(['PageNotFound'], {replaceUrl:true})
    });
  }

  getImageUrl() {
    if (this.sidenav.opened) {
      this.grippy.nativeElement.style.left = "20rem"
      return "url(../../assets/images/sidebar-grippy-hide.png)"
    } else {
      this.grippy.nativeElement.style.left = "0rem"
      return "url(../../assets/images/sidebar-grippy-show.png)"
    }
  }

  checkSidebar() {
    if (this.sidenav.opened) {
      this.grippy.nativeElement.style.backgroundImage = "url(../../assets/images/sidebar-grippy-hide.png)"
      this.grippy.nativeElement.style.left = "20rem"
    } else {
      this.grippy.nativeElement.style.backgroundImage = "url(../../assets/images/sidebar-grippy-show.png)"
      this.grippy.nativeElement.style.left = "0rem"
    }
  }

  toggleMenu() {
    this.sidenav.toggle();
    this.checkSidebar();
  }

  createRouterLink(url: string) {
    var components: Array<string> = url.substring(0, url.length - 3).split('/');
    var ret = ['/examples'];

    return ret.concat(components);
  }
}